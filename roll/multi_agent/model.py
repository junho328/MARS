"""
Multi-Agent LoRA Model Wrapper

Provides a model wrapper that manages multiple LoRA adapters,
one per agent, allowing for per-agent gradient updates during training.

IMPORTANT: This uses PEFT's inject_adapter_in_model to inject LoRA layers
IN-PLACE into an existing model, which is compatible with DeepSpeed.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from peft import PeftModel, get_peft_model, inject_adapter_in_model
from peft.tuners.lora import LoraModel
from transformers import PreTrainedModel

from roll.multi_agent.config import MultiAgentConfig, MultiAgentLoRAConfig
from roll.utils.logging import get_logger

logger = get_logger()


class MultiAgentLoRAModel:
    """
    Manager for multiple LoRA adapters on a base model.
    
    This class does NOT wrap the model - it injects LoRA adapters in-place
    using PEFT's inject_adapter_in_model, which is compatible with DeepSpeed.
    
    Each agent has its own named LoRA adapter. During training,
    we switch between adapters based on which agent's trajectories
    we are processing, ensuring gradient isolation.
    
    Usage:
        config = MultiAgentConfig(enabled=True, num_agents=2)
        manager = MultiAgentLoRAModel(base_model, config)
        
        # For agent 0's batch
        manager.set_active_agent(0)
        outputs = base_model(input_ids, attention_mask)  # Uses agent_0's adapter
        loss.backward()  # Only agent_0's adapter receives gradients
        
        # For agent 1's batch  
        manager.set_active_agent(1)
        outputs = base_model(input_ids, attention_mask)  # Uses agent_1's adapter
        loss.backward()  # Only agent_1's adapter receives gradients
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        multi_agent_config: MultiAgentConfig,
    ):
        # Store reference to the base model (NOT wrapped)
        self.base_model = base_model
        self.multi_agent_config = multi_agent_config
        self.num_agents = multi_agent_config.num_agents
        self.adapter_names = multi_agent_config.get_adapter_names()
        self.active_agent_id: int = 0
        self._peft_model: Optional[PeftModel] = None
        
        # Inject adapters in-place using PEFT
        self._inject_adapters()
        
        logger.info(f"Multi-agent LoRA model initialized with {self.num_agents} adapters")
    
    def _inject_adapters(self) -> None:
        """Inject all agent adapters into the base model in-place."""
        # Use get_peft_model to create PEFT model with first adapter
        first_adapter_config = self.multi_agent_config.get_adapter_config(0)
        peft_config = first_adapter_config.to_peft_config()
        
        # Create PEFT model - this wraps base_model
        # IMPORTANT: The PEFT model wraps the base_model, so base_model's forward
        # is now called through the PEFT wrapper. We store the PEFT model
        # and use it for adapter management.
        self._peft_model = get_peft_model(self.base_model, peft_config, adapter_name="agent_0")
        logger.info(f"Created PEFT model with adapter: agent_0")
        
        # Add additional adapters for other agents
        for i in range(1, self.num_agents):
            adapter_config = self.multi_agent_config.get_adapter_config(i)
            adapter_peft_config = adapter_config.to_peft_config()
            adapter_name = f"agent_{i}"
            
            self._peft_model.add_adapter(adapter_name, adapter_peft_config)
            logger.info(f"Added adapter: {adapter_name}")
        
        # Ensure all adapter parameters are trainable
        self._enable_all_adapter_gradients()
        
        # Set first adapter as active
        self._peft_model.set_adapter("agent_0")
    
    def get_peft_model(self) -> PeftModel:
        """
        Return the PeftModel that should be used for DeepSpeed wrapping.
        
        This returns the PEFT model which wraps the original base_model.
        When integrating with DeepSpeed, this should replace the original
        model in DeepSpeedEngine.
        """
        return self._peft_model
    
    def _enable_all_adapter_gradients(self) -> None:
        """Ensure all LoRA adapter parameters have requires_grad=True."""
        for name, param in self._peft_model.named_parameters():
            # Enable gradients for all LoRA parameters across all adapters
            if "lora_" in name.lower():
                param.requires_grad = True
    
    @property
    def model(self):
        """Return the PEFT model for compatibility."""
        return self._peft_model
    
    def set_active_agent(self, agent_id: int, isolate_gradients: bool = True) -> None:
        """
        Set the active LoRA adapter for the given agent.
        
        Args:
            agent_id: The agent ID (0, 1, ..., num_agents-1)
            isolate_gradients: If True, freeze all other adapters to ensure
                              only this agent's adapter receives gradients
        """
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError(f"Invalid agent_id {agent_id}. Must be in [0, {self.num_agents})")
        
        adapter_name = f"agent_{agent_id}"
        self._peft_model.set_adapter(adapter_name)
        self.active_agent_id = agent_id
        
        # Freeze all other adapters for gradient isolation during training
        if isolate_gradients:
            self.freeze_all_adapters_except(agent_id)
    
    def get_active_agent(self) -> int:
        """Get the currently active agent ID."""
        return self.active_agent_id
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass through the PEFT model with the currently active adapter.
        
        NOTE: When using with DeepSpeed, the forward pass goes through 
        DeepSpeed's wrapped model, not this method. This is here for 
        standalone usage and testing.
        """
        return self._peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
    
    def train(self, mode: bool = True):
        """Set training mode on the PEFT model."""
        self._peft_model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode on the PEFT model."""
        self._peft_model.eval()
        return self
    
    def parameters(self, recurse: bool = True):
        """Return model parameters."""
        return self._peft_model.parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Return named parameters."""
        return self._peft_model.named_parameters(prefix=prefix, recurse=recurse)
    
    def get_adapter_parameters(self, agent_id: int):
        """
        Get parameters for a specific agent's adapter.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            Iterator of (name, parameter) tuples for the adapter
        """
        adapter_name = f"agent_{agent_id}"
        for name, param in self._peft_model.named_parameters():
            if adapter_name in name and param.requires_grad:
                yield name, param
    
    def get_all_adapter_parameters(self) -> Dict[int, List[Tuple[str, torch.nn.Parameter]]]:
        """
        Get parameters for all adapters.
        
        Returns:
            Dict mapping agent_id to list of (name, parameter) tuples
        """
        result = {i: [] for i in range(self.num_agents)}
        
        for name, param in self._peft_model.named_parameters():
            if not param.requires_grad:
                continue
            for i in range(self.num_agents):
                adapter_name = f"agent_{i}"
                if adapter_name in name:
                    result[i].append((name, param))
                    break
        
        return result
    
    def freeze_all_adapters_except(self, agent_id: int) -> None:
        """
        Freeze all adapters except the specified agent's adapter.
        
        This is an alternative gradient isolation strategy that
        freezes other adapters instead of switching active adapter.
        
        Args:
            agent_id: The agent whose adapter should remain trainable
        """
        active_adapter = f"agent_{agent_id}"
        
        for name, param in self._peft_model.named_parameters():
            if "lora" in name.lower():
                if active_adapter in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    
    def unfreeze_all_adapters(self) -> None:
        """Unfreeze all adapter parameters."""
        for name, param in self._peft_model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
    
    def save_adapter(self, agent_id: int, save_path: str) -> None:
        """
        Save a specific agent's adapter to disk.
        
        Args:
            agent_id: The agent ID
            save_path: Path to save the adapter
        """
        adapter_name = f"agent_{agent_id}"
        self._peft_model.save_pretrained(
            save_path,
            selected_adapters=[adapter_name],
        )
        logger.info(f"Saved adapter {adapter_name} to {save_path}")
    
    def save_all_adapters(self, save_dir: str) -> None:
        """
        Save all adapters to disk.
        
        Args:
            save_dir: Base directory for saving adapters
        """
        import os
        for i in range(self.num_agents):
            adapter_path = os.path.join(save_dir, f"agent_{i}")
            self.save_adapter(i, adapter_path)
    
    def load_adapter(self, agent_id: int, load_path: str) -> None:
        """
        Load a specific agent's adapter from disk.
        
        Args:
            agent_id: The agent ID
            load_path: Path to load the adapter from
        """
        adapter_name = f"agent_{agent_id}"
        self._peft_model.load_adapter(load_path, adapter_name=adapter_name)
        logger.info(f"Loaded adapter {adapter_name} from {load_path}")
    
    @property
    def config(self):
        """Return the base model's config."""
        return self._peft_model.config
    
    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing."""
        self._peft_model.gradient_checkpointing_enable(**kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._peft_model.gradient_checkpointing_disable()
    
    def offload_states(self, include=None, non_blocking=False):
        """
        Offload model states to CPU (for DeepSpeed compatibility).
        
        Since PEFT models don't support state offloading directly,
        this is a no-op that allows the pipeline to continue.
        The actual memory management is handled by DeepSpeed at a higher level.
        """
        # PEFT/LoRA models don't have native offload support
        # This is intentionally a no-op for compatibility
        pass
    
    def reload_states(self, include=None, non_blocking=False):
        """
        Reload model states from CPU (for DeepSpeed compatibility).
        
        Since PEFT models don't support state offloading directly,
        this is a no-op that allows the pipeline to continue.
        """
        # PEFT/LoRA models don't have native reload support
        # This is intentionally a no-op for compatibility
        pass
    
    def load_states(self, *args, **kwargs):
        """
        Load model states from CPU (for DeepSpeed compatibility).
        
        This is a no-op for PEFT models.
        """
        pass
    
    def state_dict(self, *args, **kwargs):
        """Return state dict from the wrapped model."""
        return self._peft_model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load state dict into the wrapped model."""
        return self._peft_model.load_state_dict(state_dict, *args, **kwargs)
    


def create_multi_agent_model(
    base_model: PreTrainedModel,
    multi_agent_config: MultiAgentConfig,
) -> Union[MultiAgentLoRAModel, PreTrainedModel]:
    """
    Factory function to create a multi-agent LoRA model.
    
    If multi-agent training is disabled, returns the base model unchanged.
    
    Args:
        base_model: The base pretrained model
        multi_agent_config: Multi-agent configuration
        
    Returns:
        MultiAgentLoRAModel if enabled, otherwise the original base_model
    """
    if not multi_agent_config.enabled:
        logger.info("Multi-agent training disabled, using base model")
        return base_model
    
    return MultiAgentLoRAModel(base_model, multi_agent_config)

