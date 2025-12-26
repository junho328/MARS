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
from roll.multi_agent.model import MultiAgentLoRAModel, create_multi_agent_model
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
    1. Wraps base model with MultiAgentLoRAModel on initialization
    2. Partitions training batches by agent_id
    3. Switches active LoRA adapter before processing each agent's batch
    4. Accumulates per-agent metrics separately
    """
    
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config)
        self.multi_agent_config: Optional[MultiAgentConfig] = None
        self.multi_agent_model: Optional[MultiAgentLoRAModel] = None
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_multi_agent_config(self, config: MultiAgentConfig):
        """
        Set the multi-agent configuration and wrap the model.
        
        This should be called after initialize() but before training.
        """
        self.multi_agent_config = config
        
        if not config.enabled:
            logger.info("Multi-agent training disabled, using standard model")
            return
        
        logger.info(f"Configuring MultiAgentActorWorker with {config.num_agents} agents")
        
        # Get the base model from strategy and wrap with MultiAgentLoRAModel
        base_model = self.strategy.unwrap_model()
        
        # Wrap with multi-agent LoRA
        self.multi_agent_model = create_multi_agent_model(base_model, config)
        
        # Replace the model in the strategy
        # This depends on the strategy implementation
        if hasattr(self.strategy, 'model'):
            self.strategy.model = self.multi_agent_model
        
        logger.info(f"MultiAgentActorWorker configured with {config.num_agents} agents")
    
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
        
        Args:
            data: The full training batch
            
        Returns:
            Dict mapping agent_id to DataProto containing that agent's samples
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
        
        This method works with both MultiAgentLoRAModel and standard models.
        """
        model = self.strategy.unwrap_model()
        
        if hasattr(model, 'set_active_agent'):
            # MultiAgentLoRAModel
            model.set_active_agent(agent_id)
        elif hasattr(model, 'set_adapter'):
            # Standard PEFT model with multiple adapters
            adapter_name = f"agent_{agent_id}"
            model.set_adapter(adapter_name)
        else:
            # Standard model without adapter switching
            logger.warning(
                f"Model does not support adapter switching. "
                f"Ignoring set_active_agent({agent_id})"
            )
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def do_checkpoint(self, global_step):
        """
        Save checkpoint for multi-agent model.
        
        Saves each agent's adapter separately.
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
            
            # Check if model supports multi-agent saving
            model = self.strategy.unwrap_model()
            
            if hasattr(model, 'save_all_adapters'):
                model.save_all_adapters(save_dir)
                exec_metrics = {"adapters_saved": self.multi_agent_config.num_agents}
            else:
                # Fall back to standard checkpoint
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
    Get the appropriate worker class for multi-agent training.
    
    Returns the worker class path string for use in configuration.
    """
    return "roll.multi_agent.worker.MultiAgentActorWorker"

