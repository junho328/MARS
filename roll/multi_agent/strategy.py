"""
Multi-Agent DeepSpeed Strategy

Custom DeepSpeed strategy that applies PEFT adapters BEFORE DeepSpeed initialization,
ensuring proper integration between PEFT and DeepSpeed.
"""

from datetime import timedelta
from typing import Optional

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from transformers import get_scheduler, set_seed

from peft import get_peft_model

from roll.distributed.strategy.deepspeed_strategy import DeepSpeedTrainStrategy
from roll.models.model_providers import default_tokenizer_provider, default_processor_provider
from roll.multi_agent.config import MultiAgentConfig, MultiAgentLoRAConfig
from roll.third_party.deepspeed.offload_states_patch import bind_deepspeed_offload_states_func
from roll.utils.deepspeed_utils import get_optimizer_grouped_parameters
from roll.utils.logging import get_logger

logger = get_logger()


class MultiAgentDeepSpeedStrategy(DeepSpeedTrainStrategy):
    """
    DeepSpeed training strategy with multi-agent LoRA support.
    
    This strategy applies PEFT adapters to the model BEFORE passing it
    to deepspeed.initialize(), ensuring proper integration.
    
    The multi-agent config should be set via set_multi_agent_config()
    before calling initialize().
    """
    
    strategy_name = "multi_agent_deepspeed_train"
    
    def __init__(self, worker):
        super().__init__(worker)
        self.multi_agent_config: Optional[MultiAgentConfig] = None
        self._peft_model = None
    
    def set_multi_agent_config(self, config: MultiAgentConfig):
        """Set the multi-agent configuration before initialization."""
        self.multi_agent_config = config
        logger.info(f"MultiAgentDeepSpeedStrategy configured with {config.num_agents} agents")
    
    def initialize(self, model_provider):
        """
        Initialize the strategy with PEFT applied before DeepSpeed.
        
        This overrides the parent's initialize to inject PEFT adapters
        between model creation and DeepSpeed wrapping.
        """
        assert self.ds_config._stage > 0, "deepspeed train only supports zero > 0."
        
        set_seed(seed=self.worker.pipeline_config.seed)
        deepspeed.init_distributed(timeout=timedelta(minutes=self.worker_config.backend_timeout))
        dist.all_reduce(torch.zeros(1).cuda())
        
        self.worker.rank_info.dp_rank = dist.get_rank()
        self.worker.rank_info.dp_size = dist.get_world_size()
        
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.processor = default_processor_provider(model_args=self.worker_config.model_args)
        
        # Get the base model
        model = model_provider(
            tokenizer=self.tokenizer, 
            model_args=self.worker_config.model_args, 
            is_trainable=True
        )
        
        # Apply PEFT adapters BEFORE DeepSpeed initialization
        if self.multi_agent_config is not None and self.multi_agent_config.enabled:
            model = self._apply_multi_agent_lora(model)
            logger.info("Applied multi-agent LoRA adapters before DeepSpeed init")
        
        # Create optimizer with the PEFT model's parameters
        adam_optimizer = DeepSpeedCPUAdam if self.ds_config.is_offload() else FusedAdam
        optim_params = get_optimizer_grouped_parameters(
            model, weight_decay=self.worker_config.training_args.weight_decay
        )
        optimizer = adam_optimizer(
            optim_params,
            lr=self.worker_config.training_args.learning_rate,
            betas=(self.worker_config.training_args.adam_beta1, self.worker_config.training_args.adam_beta2),
        )
        
        logger.info(f"max steps pipeline {self.worker_config.training_args.max_steps}")
        self.worker_config.training_args.max_steps = (
            self.worker_config.training_args.max_steps // self.worker.rank_info.dp_size
        )
        logger.info(f"max steps worker train {self.worker_config.training_args.max_steps}")
        
        scheduler = get_scheduler(
            self.worker_config.training_args.lr_scheduler_type,
            optimizer,
            num_warmup_steps=self.worker_config.training_args.get_warmup_steps(
                self.worker_config.training_args.max_steps
            ),
            num_training_steps=self.worker_config.training_args.max_steps,
            scheduler_specific_kwargs={
                "min_lr": 0.0,
            },
        )
        
        # Initialize DeepSpeed with the PEFT model
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model_parameters=model.parameters(),
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=self.worker_config.strategy_args.strategy_config,
            dist_init_required=True,
        )
        bind_deepspeed_offload_states_func(self.model)
        
        logger.info(f"{self.model}")
        dist.barrier()
    
    def _apply_multi_agent_lora(self, base_model):
        """
        Apply multi-agent LoRA adapters to the base model.
        
        Creates a PeftModel with multiple named adapters, one per agent.
        """
        config = self.multi_agent_config
        
        # Create PEFT config for first adapter
        first_adapter_config = config.get_adapter_config(0)
        peft_config = first_adapter_config.to_peft_config()
        
        # Create PEFT model with first adapter
        peft_model = get_peft_model(base_model, peft_config, adapter_name="agent_0")
        logger.info(f"Created PEFT model with adapter: agent_0")
        
        # Add additional adapters for other agents
        for i in range(1, config.num_agents):
            adapter_config = config.get_adapter_config(i)
            adapter_peft_config = adapter_config.to_peft_config()
            adapter_name = f"agent_{i}"
            
            peft_model.add_adapter(adapter_name, adapter_peft_config)
            logger.info(f"Added adapter: {adapter_name}")
        
        # Enable gradients for all LoRA parameters
        for name, param in peft_model.named_parameters():
            if "lora_" in name.lower():
                param.requires_grad = True
        
        # Set first adapter as active
        peft_model.set_adapter("agent_0")
        
        # Store reference for later use
        self._peft_model = peft_model
        
        logger.info(f"Multi-agent LoRA model initialized with {config.num_agents} adapters")
        return peft_model
    
    def get_peft_model(self):
        """Return the PEFT model for adapter management."""
        return self._peft_model
    
    def set_active_agent(self, agent_id: int, isolate_gradients: bool = True):
        """
        Set the active LoRA adapter for the given agent.
        
        Args:
            agent_id: The agent ID (0, 1, ..., num_agents-1)
            isolate_gradients: If True, freeze other adapters
        """
        if self._peft_model is None:
            logger.warning("No PEFT model available for adapter switching")
            return
        
        adapter_name = f"agent_{agent_id}"
        self._peft_model.set_adapter(adapter_name)
        
        if isolate_gradients:
            self._freeze_all_adapters_except(agent_id)
    
    def _freeze_all_adapters_except(self, active_agent_id: int):
        """Freeze all adapter parameters except the active one."""
        if self._peft_model is None:
            return
        
        active_adapter = f"agent_{active_agent_id}"
        
        for name, param in self._peft_model.named_parameters():
            if "lora_" in name.lower():
                # Check if this parameter belongs to the active adapter
                if active_adapter in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    
    def unfreeze_all_adapters(self):
        """Re-enable gradients for all adapter parameters."""
        if self._peft_model is None:
            return
        
        for name, param in self._peft_model.named_parameters():
            if "lora_" in name.lower():
                param.requires_grad = True
    
    def _strip_peft_prefix(self, param_name: str) -> str:
        """
        Strip PEFT prefix from parameter names for vLLM compatibility.
        
        PEFT wraps the model with prefixes like:
        - base_model.model.layers... -> model.layers...
        - base_model.model.embed_tokens... -> model.embed_tokens...
        
        PEFT also wraps LoRA target layers with .base_layer.:
        - layers.0.self_attn.q_proj.base_layer.weight -> model.layers.0.self_attn.q_proj.weight
        
        Also handles lm_head which may be:
        - base_model.model.lm_head -> lm_head  
        """
        # Skip LoRA adapter parameters - they shouldn't be synced to vLLM
        if "lora_" in param_name.lower():
            return None
        
        result = param_name
        
        # Strip PEFT wrapper prefixes
        if result.startswith("base_model.model."):
            result = result[len("base_model.model."):]
        elif result.startswith("base_model."):
            result = result[len("base_model."):]
        
        # Remove .base_layer. from LoRA wrapped layers
        # e.g., layers.0.self_attn.q_proj.base_layer.weight -> layers.0.self_attn.q_proj.weight
        result = result.replace(".base_layer.", ".")
        
        # Add back 'model.' prefix for vLLM compatibility
        # vLLM expects: model.layers.0.self_attn.q_proj.weight
        if not result.startswith("model.") and not result.startswith("lm_head"):
            result = "model." + result
        
        return result
    
    def model_update(self, tgt_workers, broadcast_tgt_devices, p2p_tgt_devices):
        """
        Override model_update to handle PEFT parameter name mapping.
        
        Strips the PEFT prefix from parameter names so they match
        vLLM's expected parameter structure.
        """
        from codetiming import Timer
        from tqdm import tqdm
        import ray
        from deepspeed.runtime.zero import GatheredParameters
        from roll.utils.collective import collective
        
        comm_plan = self.model_update_comm_plan[self.worker.rank_info.pp_rank]
        model = self.unwrap_model()
        
        with Timer("model_update_total") as timer_total:
            params_list = list(model.named_parameters())
            for param_name, param in tqdm(
                params_list, 
                desc="weight update progress", 
                total=len(params_list)
            ):
                # Map PEFT parameter names to vLLM names
                vllm_param_name = self._strip_peft_prefix(param_name)
                
                # Skip LoRA adapter parameters
                if vllm_param_name is None:
                    continue
                
                shape = param.shape if not self.ds_config.is_zero3() else param.ds_shape
                
                if not self.ds_config.is_zero3():
                    param_weight = param.data
                    refs = []
                    
                    for p2p_tgt_device in p2p_tgt_devices:
                        p2p_tgt_worker = tgt_workers[p2p_tgt_device["rank"]]
                        ref = p2p_tgt_worker.update_parameter.remote(
                            parameter_name=vllm_param_name,
                            weight=param_weight,
                            ranks_in_worker=[p2p_tgt_device["device"]["rank"]],
                        )
                        refs.append(ref)
                    
                    if (
                        self.worker.rank_info.tp_rank == 0
                        and self.worker.rank_info.cp_rank == 0
                        and self.worker.rank_info.dp_rank == 0
                    ):
                        for worker in tgt_workers:
                            ref = worker.broadcast_parameter.remote(
                                src_pp_rank=self.worker.rank_info.pp_rank,
                                dtype=param_weight.dtype,
                                shape=shape,
                                parameter_name=vllm_param_name,
                            )
                            refs.append(ref)
                    
                    if len(broadcast_tgt_devices) > 0:
                        collective.broadcast(
                            tensor=param_weight, 
                            src_rank=0, 
                            group_name=comm_plan["group_name"]
                        )
                    ray.get(refs)
                
                else:
                    with GatheredParameters([param]):
                        param_weight = param.data
                        refs = []
                        
                        for p2p_tgt_device in p2p_tgt_devices:
                            p2p_tgt_worker = tgt_workers[p2p_tgt_device["rank"]]
                            ref = p2p_tgt_worker.update_parameter.remote(
                                parameter_name=vllm_param_name,
                                weight=param_weight,
                                ranks_in_worker=[p2p_tgt_device["device"]["rank"]],
                            )
                            refs.append(ref)
                        
                        if (
                            self.worker.rank_info.tp_rank == 0
                            and self.worker.rank_info.cp_rank == 0
                            and self.worker.rank_info.dp_rank == 0
                        ):
                            for worker in tgt_workers:
                                ref = worker.broadcast_parameter.remote(
                                    src_pp_rank=self.worker.rank_info.pp_rank,
                                    dtype=param_weight.dtype,
                                    shape=shape,
                                    parameter_name=vllm_param_name,
                                )
                                refs.append(ref)
                        
                        if len(broadcast_tgt_devices) > 0:
                            collective.broadcast(
                                tensor=param_weight, 
                                src_rank=0, 
                                group_name=comm_plan["group_name"]
                            )
                        ray.get(refs)
        
        return {"total": timer_total.last}


# Register the strategy
def register_multi_agent_strategy():
    """Register the multi-agent strategy in the factory."""
    from roll.distributed.strategy.factory import STRATEGY_REGISTRY
    STRATEGY_REGISTRY["multi_agent_deepspeed_train"] = MultiAgentDeepSpeedStrategy

