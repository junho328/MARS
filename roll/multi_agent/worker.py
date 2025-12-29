"""
Multi-Agent Actor Worker

Modified actor worker that handles per-agent LoRA adapter training.
Routes training batches to the correct adapter based on agent_id.
"""

import os
from typing import Dict, List, Optional, Union

import ray
import torch
from tqdm import tqdm

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import register, Dispatch
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import TrainStrategy
from roll.models.model_providers import default_actor_model_provider
from roll.multi_agent.config import MultiAgentConfig
from roll.pipeline.base_worker import ActorWorker
from roll.utils.context_managers import state_offload_manger
from roll.utils.functionals import append_to_dict, agg_loss, compute_approx_kl, masked_mean
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType

logger = get_logger()


class MultiAgentActorWorker(ActorWorker):
    """
    Actor worker modified for multi-agent LoRA training.
    
    Key differences from ActorWorker:
    1. Uses MultiAgentDeepSpeedStrategy that applies PEFT before DeepSpeed
    2. Partitions training batches by agent_id
    3. Switches active LoRA adapter before processing each agent's batch
    4. Accumulates per-agent metrics separately
    """
    
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config)
        self.multi_agent_config: Optional[MultiAgentConfig] = None
        self._pending_multi_agent_config: Optional[MultiAgentConfig] = None
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_multi_agent_config_before_init(self, config: MultiAgentConfig):
        """
        Set the multi-agent configuration BEFORE initialize() is called.
        
        This stores the config so it can be passed to the strategy during initialization.
        """
        self._pending_multi_agent_config = config
        logger.info(f"Stored multi-agent config for initialization: {config.num_agents} agents")
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        """
        Initialize the worker with multi-agent support.
        
        This overrides the parent's initialize to pass multi-agent config
        to the strategy BEFORE calling strategy.initialize().
        """
        # Call Worker's initialize (skipping ActorWorker's)
        Worker.initialize(self, pipeline_config)
        
        # Create the strategy
        self.strategy = create_strategy(worker=self)
        
        # If we have a pending multi-agent config, pass it to the strategy
        if self._pending_multi_agent_config is not None:
            self.multi_agent_config = self._pending_multi_agent_config
            
            # Check if strategy supports multi-agent config
            if hasattr(self.strategy, 'set_multi_agent_config'):
                self.strategy.set_multi_agent_config(self.multi_agent_config)
                logger.info(f"Passed multi-agent config to strategy")
            else:
                logger.warning(
                    f"Strategy {type(self.strategy).__name__} does not support "
                    f"set_multi_agent_config. Multi-agent LoRA will not be applied."
                )
        
        # Now initialize the strategy (this will apply PEFT before DeepSpeed)
        self.strategy.initialize(model_provider=default_actor_model_provider)
        self.tokenizer = self.strategy.tokenizer
        
        if self.pipeline_config.resume_from_checkpoint:
            load_dir = self.pipeline_config.resume_from_checkpoint
            self.strategy.load_checkpoint(load_dir=load_dir, tag="checkpoint")
        
        self.logger.info(f"{self.worker_name} initialized with multi-agent support")
        
        self.strategy.offload_states()
        
        # Initialize CUDA context
        torch.cuda.init()
    
    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def train_step(self, data: DataProto):
        """
        Training step with per-agent adapter routing.
        
        Partitions the batch by agent_id and trains each agent's
        adapter separately with gradient isolation.
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        
        self.logger.info(f"{self.worker_name} multi-agent train step {global_step}")
        
        # Check if multi-agent training is enabled
        if self.multi_agent_config is None or not self.multi_agent_config.enabled:
            # Fall back to standard training
            return super().train_step(data)
        
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/train_step",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params, OffloadStateType.other_params]},
        ):
            data = data.to("cuda")
            data = self.strategy.get_data_input(data)
            
            per_device_train_batch_size = self.worker_config.training_args.per_device_train_batch_size
            backward_batch_size = (
                per_device_train_batch_size * self.worker_config.training_args.gradient_accumulation_steps
            )
            
            # Partition batch by agent_id
            agent_batches = self._partition_by_agent(data)
            
            # Train each agent's adapter sequentially
            for agent_id, agent_data in agent_batches.items():
                if agent_data is None or agent_data.batch is None:
                    continue
                
                batch_size = agent_data.batch.batch_size[0]
                if batch_size == 0:
                    continue
                
                self.logger.info(
                    f"Training agent_{agent_id} with {batch_size} samples"
                )
                
                # Switch to this agent's adapter
                self._set_active_agent(agent_id)
                
                # Create dataloader for this agent's batch
                dataloader = agent_data.make_iterator(
                    mini_batch_size=backward_batch_size,
                    epochs=self.pipeline_config.ppo_epochs,
                    dataloader_kwargs={"shuffle": True},
                )
                
                num_batches = max(1, batch_size * self.pipeline_config.ppo_epochs // backward_batch_size)
                
                for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=f"{self.worker_name} agent_{agent_id} step {global_step}",
                    total=num_batches,
                ):
                    pg_metrics = self.strategy.train_step(batch=batch, loss_func=self.loss_func)
                    
                    # Prefix metrics with agent ID
                    agent_metrics = {
                        f"agent_{agent_id}/{k}": v for k, v in pg_metrics.items()
                    }
                    append_to_dict(metrics, agent_metrics)
            
            metrics["actor/lr"] = self.strategy.scheduler.get_last_lr()[0]
            data.to("cpu")
        
        output = DataProto(meta_info={"metrics": metrics})
        return output
    
    def _partition_by_agent(self, data: DataProto) -> Dict[int, DataProto]:
        """
        Partition a batch by agent_id.
        
        Uses group_ids to determine agent assignment. In self-play,
        group_ids have format "{base}_p{player_id}".
        """
        result = {}
        
        if "group_ids" not in data.non_tensor_batch:
            # No group_ids, assign all to agent 0
            logger.warning("No group_ids in batch, assigning all to agent_0")
            result[0] = data
            return result
        
        group_ids = data.non_tensor_batch["group_ids"]
        batch_size = len(group_ids)
        
        # Determine agent_id for each sample
        agent_ids = []
        for gid in group_ids:
            agent_id = self.multi_agent_config.agent_id_from_group_id(str(gid))
            agent_ids.append(agent_id)
        
        # Create mask for each agent
        for agent_id in range(self.multi_agent_config.num_agents):
            indices = [i for i, aid in enumerate(agent_ids) if aid == agent_id]
            
            if len(indices) == 0:
                result[agent_id] = None
                continue
            
            # Extract samples for this agent
            indices_tensor = torch.tensor(indices, device=data.batch.device)
            agent_data = data.select(indices=indices_tensor)
            result[agent_id] = agent_data
            
            logger.debug(f"Agent {agent_id}: {len(indices)} samples")
        
        return result
    
    def _set_active_agent(self, agent_id: int) -> None:
        """
        Set the active LoRA adapter for the given agent.
        """
        # Use strategy's method if available (MultiAgentDeepSpeedStrategy)
        if hasattr(self.strategy, 'set_active_agent'):
            self.strategy.set_active_agent(agent_id)
            return
        
        # Fallback: try to access model directly
        model = self.strategy.unwrap_model()
        
        if hasattr(model, 'set_adapter'):
            # PEFT model
            adapter_name = f"agent_{agent_id}"
            model.set_adapter(adapter_name)
        else:
            logger.warning(
                f"Cannot switch adapter. Model does not support set_adapter()."
            )
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def do_checkpoint(self, global_step):
        """
        Save checkpoint for multi-agent model.
        """
        from codetiming import Timer
        
        with Timer("do_checkpoint") as total_timer:
            ckpt_id = f"checkpoint-{global_step}"
            save_dir = os.path.join(
                self.pipeline_config.output_dir, 
                self.worker_name, 
                ckpt_id
            )
            
            self.logger.info(f"Saving multi-agent checkpoint to {save_dir}")
            
            # Use strategy's save if available
            exec_metrics = self.strategy.save_checkpoint(save_dir, global_step, ckpt_id)
        
        metrics = {
            f"time/{self.cluster_name}/do_checkpoint/total": total_timer.last,
        }
        metric_prefix = f"time/{self.cluster_name}/do_checkpoint"
        metrics.update({f"{metric_prefix}/{k}": v for k, v in exec_metrics.items()})
        
        output = DataProto(meta_info={"metrics": metrics})
        return output


def get_multi_agent_worker_class():
    """
    Get the worker class path for configuration.
    """
    return "roll.multi_agent.worker.MultiAgentActorWorker"
