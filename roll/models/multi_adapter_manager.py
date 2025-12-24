"""
Multi-Adapter Manager for Per-Agent LoRA Adapters

This module provides functionality to create, manage, and switch between
multiple LoRA adapters, one per agent in multi-agent RL settings.

Key Features:
- Create N LoRA adapters for N agents
- Activate/deactivate adapters by agent ID
- Gradient isolation: only active adapter receives gradients
- Save/load adapters independently

Usage:
    manager = MultiAdapterManager(model, num_agents=2, lora_config=config)
    manager.activate_adapter(agent_id=0)
    output = model(inputs)  # Uses adapter_0
    loss.backward()  # Gradients only go to adapter_0
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import PreTrainedModel

from roll.utils.logging import get_logger

logger = get_logger()


@dataclass
class AdapterConfig:
    """Configuration for LoRA adapters."""
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    target_modules: List[str] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def to_peft_config(self) -> LoraConfig:
        """Convert to PEFT LoraConfig."""
        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "TOKEN_CLS": TaskType.TOKEN_CLS,
            "SEQ_CLS": TaskType.SEQ_CLS,
        }
        return LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=task_type_map.get(self.task_type, TaskType.CAUSAL_LM),
        )


class MultiAdapterManager:
    """
    Manages multiple LoRA adapters for multi-agent training.
    
    Each agent gets a dedicated adapter named 'adapter_{agent_id}'.
    Only one adapter is active at a time, and gradients only flow
    to the active adapter during backpropagation.
    
    Attributes:
        model: The base model with PEFT adapters attached
        num_agents: Number of agents/adapters
        adapter_names: List of adapter names
        active_adapter: Currently active adapter name
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        num_agents: int,
        adapter_config: Union[AdapterConfig, Dict],
        freeze_base_model: bool = True,
    ):
        """
        Initialize the multi-adapter manager.
        
        Args:
            model: Base pretrained model
            num_agents: Number of agents (each gets one adapter)
            adapter_config: LoRA configuration (AdapterConfig or dict)
            freeze_base_model: Whether to freeze base model weights
        """
        self.num_agents = num_agents
        self.adapter_names = [f"adapter_{i}" for i in range(num_agents)]
        self.active_adapter: Optional[str] = None
        self._freeze_base_model = freeze_base_model
        
        # Convert dict to AdapterConfig if needed
        if isinstance(adapter_config, dict):
            adapter_config = AdapterConfig(**adapter_config)
        self.adapter_config = adapter_config
        
        # Create adapters
        self.model = self._create_adapters(model)
        
        # Freeze base model if requested
        if freeze_base_model:
            self._freeze_base_weights()
        
        logger.info(
            f"MultiAdapterManager initialized with {num_agents} adapters: {self.adapter_names}"
        )
    
    def _create_adapters(self, model: PreTrainedModel) -> PeftModel:
        """
        Create N LoRA adapters on the model.
        
        The first adapter is created with get_peft_model, subsequent
        adapters are added with add_adapter.
        """
        peft_config = self.adapter_config.to_peft_config()
        
        # Create first adapter
        first_adapter_name = self.adapter_names[0]
        peft_model = get_peft_model(model, peft_config, adapter_name=first_adapter_name)
        logger.info(f"Created primary adapter: {first_adapter_name}")
        
        # Add additional adapters
        for adapter_name in self.adapter_names[1:]:
            peft_model.add_adapter(adapter_name, peft_config)
            logger.info(f"Added adapter: {adapter_name}")
        
        # Set first adapter as active by default
        peft_model.set_adapter(first_adapter_name)
        self.active_adapter = first_adapter_name
        
        return peft_model
    
    def _freeze_base_weights(self):
        """Freeze all base model weights, keeping only adapters trainable."""
        for name, param in self.model.named_parameters():
            # Only adapter parameters should be trainable
            if "lora_" not in name:
                param.requires_grad = False
        
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
    
    def get_adapter_name(self, agent_id: int) -> str:
        """Get adapter name for a given agent ID."""
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError(
                f"agent_id {agent_id} out of range [0, {self.num_agents})"
            )
        return self.adapter_names[agent_id]
    
    def activate_adapter(self, agent_id: int, enable_gradient: bool = True):
        """
        Activate the adapter for the specified agent.
        
        This sets the adapter as active and optionally enables gradients
        only for this adapter (disabling gradients for all others).
        
        Args:
            agent_id: The agent whose adapter should be activated
            enable_gradient: If True, only enable gradients for this adapter
        """
        adapter_name = self.get_adapter_name(agent_id)
        
        if adapter_name == self.active_adapter:
            return  # Already active
        
        # Set the adapter
        self.model.set_adapter(adapter_name)
        self.active_adapter = adapter_name
        
        if enable_gradient:
            self._set_adapter_gradients(agent_id)
        
        logger.debug(f"Activated adapter: {adapter_name}")
    
    def _set_adapter_gradients(self, active_agent_id: int):
        """
        Enable gradients only for the specified agent's adapter.
        
        All other adapters have their gradients disabled to ensure
        gradient isolation during backpropagation.
        """
        active_adapter = self.get_adapter_name(active_agent_id)
        
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                # Check if this parameter belongs to the active adapter
                # PEFT names adapters like: base_model.model.layers.0.self_attn.q_proj.lora_A.adapter_0.weight
                is_active = active_adapter in name
                param.requires_grad = is_active
    
    def disable_all_adapters(self):
        """Disable all adapters (use base model only)."""
        self.model.disable_adapters()
        self.active_adapter = None
        logger.debug("All adapters disabled")
    
    def enable_adapters(self):
        """Re-enable adapters after disable_all_adapters."""
        self.model.enable_adapters()
        if self.active_adapter is None:
            self.active_adapter = self.adapter_names[0]
            self.model.set_adapter(self.active_adapter)
        logger.debug(f"Adapters enabled, active: {self.active_adapter}")
    
    def get_active_agent_id(self) -> Optional[int]:
        """Get the agent ID of the currently active adapter."""
        if self.active_adapter is None:
            return None
        return int(self.active_adapter.split("_")[1])
    
    def save_adapters(self, save_dir: str, adapter_ids: Optional[List[int]] = None):
        """
        Save adapters to disk.
        
        Args:
            save_dir: Directory to save adapters
            adapter_ids: List of agent IDs to save (None = all)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if adapter_ids is None:
            adapter_ids = list(range(self.num_agents))
        
        for agent_id in adapter_ids:
            adapter_name = self.get_adapter_name(agent_id)
            adapter_path = os.path.join(save_dir, adapter_name)
            
            # Temporarily set this adapter as active to save it
            self.model.set_adapter(adapter_name)
            self.model.save_pretrained(adapter_path)
            
            logger.info(f"Saved adapter {adapter_name} to {adapter_path}")
        
        # Restore previously active adapter
        if self.active_adapter:
            self.model.set_adapter(self.active_adapter)
    
    def load_adapters(self, load_dir: str, adapter_ids: Optional[List[int]] = None):
        """
        Load adapters from disk.
        
        Args:
            load_dir: Directory containing saved adapters
            adapter_ids: List of agent IDs to load (None = all)
        """
        if adapter_ids is None:
            adapter_ids = list(range(self.num_agents))
        
        for agent_id in adapter_ids:
            adapter_name = self.get_adapter_name(agent_id)
            adapter_path = os.path.join(load_dir, adapter_name)
            
            if not os.path.exists(adapter_path):
                logger.warning(f"Adapter path not found: {adapter_path}")
                continue
            
            # Load adapter weights
            self.model.load_adapter(adapter_path, adapter_name=adapter_name)
            logger.info(f"Loaded adapter {adapter_name} from {adapter_path}")
        
        # Restore previously active adapter
        if self.active_adapter:
            self.model.set_adapter(self.active_adapter)
    
    def get_adapter_state_dict(self, agent_id: int) -> Dict[str, torch.Tensor]:
        """Get state dict for a specific adapter."""
        adapter_name = self.get_adapter_name(agent_id)
        state_dict = {}
        
        for name, param in self.model.named_parameters():
            if adapter_name in name and "lora_" in name:
                state_dict[name] = param.data.clone()
        
        return state_dict
    
    def set_adapter_state_dict(
        self, agent_id: int, state_dict: Dict[str, torch.Tensor]
    ):
        """Set state dict for a specific adapter."""
        adapter_name = self.get_adapter_name(agent_id)
        
        model_state = self.model.state_dict()
        for name, param in state_dict.items():
            if adapter_name in name:
                model_state[name] = param
        
        self.model.load_state_dict(model_state, strict=False)
    
    def merge_and_unload(self, agent_id: Optional[int] = None) -> PreTrainedModel:
        """
        Merge adapter weights into base model and return unloaded model.
        
        Args:
            agent_id: If specified, merge only this agent's adapter.
                     If None, merge the currently active adapter.
        
        Returns:
            The base model with adapter weights merged in
        """
        if agent_id is not None:
            self.activate_adapter(agent_id, enable_gradient=False)
        
        return self.model.merge_and_unload()
    
    def print_adapter_info(self):
        """Print information about all adapters."""
        print(f"Number of agents: {self.num_agents}")
        print(f"Adapter names: {self.adapter_names}")
        print(f"Active adapter: {self.active_adapter}")
        print(f"LoRA config: rank={self.adapter_config.lora_rank}, "
              f"alpha={self.adapter_config.lora_alpha}, "
              f"targets={self.adapter_config.target_modules}")
        
        # Count parameters per adapter
        for adapter_name in self.adapter_names:
            adapter_params = sum(
                p.numel() for name, p in self.model.named_parameters()
                if adapter_name in name and "lora_" in name
            )
            print(f"  {adapter_name}: {adapter_params:,} parameters")


def create_multi_adapter_model(
    model: PreTrainedModel,
    num_agents: int,
    lora_target: str = "q_proj,k_proj,v_proj,o_proj",
    lora_rank: int = 32,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    freeze_base_model: bool = True,
) -> MultiAdapterManager:
    """
    Convenience function to create a multi-adapter model.
    
    Args:
        model: Base pretrained model
        num_agents: Number of agents
        lora_target: Comma-separated target modules
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        freeze_base_model: Whether to freeze base model
    
    Returns:
        MultiAdapterManager instance
    """
    target_modules = [t.strip() for t in lora_target.split(",")]
    
    adapter_config = AdapterConfig(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    
    return MultiAdapterManager(
        model=model,
        num_agents=num_agents,
        adapter_config=adapter_config,
        freeze_base_model=freeze_base_model,
    )

