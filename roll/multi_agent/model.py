"""
Multi-Agent LoRA Model Wrapper

Provides a model wrapper that manages multiple LoRA adapters,
one per agent, allowing for per-agent gradient updates during training.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from peft import PeftModel, get_peft_model
from transformers import PreTrainedModel

from roll.multi_agent.config import MultiAgentConfig, MultiAgentLoRAConfig
from roll.utils.logging import get_logger

logger = get_logger()


class MultiAgentLoRAModel(nn.Module):
    """
    Wrapper for a base model with multiple LoRA adapters.
    
    Each agent has its own named LoRA adapter. During training,
    we switch between adapters based on which agent's trajectories
    we are processing, ensuring gradient isolation.
    
    Usage:
        config = MultiAgentConfig(enabled=True, num_agents=2)
        model = MultiAgentLoRAModel(base_model, config)
        
        # For agent 0's batch
        model.set_active_agent(0)
        outputs = model(input_ids, attention_mask)
        loss.backward()  # Only agent_0's adapter receives gradients
        
        # For agent 1's batch
        model.set_active_agent(1)
        outputs = model(input_ids, attention_mask)
        loss.backward()  # Only agent_1's adapter receives gradients
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        multi_agent_config: MultiAgentConfig,
    ):
        super().__init__()
        
        self.multi_agent_config = multi_agent_config
        self.num_agents = multi_agent_config.num_agents
        self.adapter_names = multi_agent_config.get_adapter_names()
        self.active_agent_id: int = 0
        
        # Initialize PEFT model with the first adapter
        first_adapter_config = multi_agent_config.get_adapter_config(0)
        peft_config = first_adapter_config.to_peft_config()
        
        self.model = get_peft_model(base_model, peft_config, adapter_name="agent_0")
        logger.info(f"Created PEFT model with adapter: agent_0")
        
        # Add additional adapters for other agents
        for i in range(1, self.num_agents):
            adapter_config = multi_agent_config.get_adapter_config(i)
            adapter_peft_config = adapter_config.to_peft_config()
            adapter_name = f"agent_{i}"
            
            self.model.add_adapter(adapter_name, adapter_peft_config)
            logger.info(f"Added adapter: {adapter_name}")
        
        # Ensure all adapter parameters are trainable
        self._enable_all_adapter_gradients()
        
        # Set first adapter as active
        self.model.set_adapter("agent_0")
        logger.info(f"Multi-agent LoRA model initialized with {self.num_agents} adapters")
    
    def _enable_all_adapter_gradients(self) -> None:
        """Ensure all LoRA adapter parameters have requires_grad=True."""
        for name, param in self.model.named_parameters():
            # Enable gradients for all LoRA parameters across all adapters
            if "lora_" in name.lower():
                param.requires_grad = True
    
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
        self.model.set_adapter(adapter_name)
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
        Forward pass through the model with the currently active adapter.
        
        Returns the output from the active agent's adapter.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.model.eval()
        return self
    
    def parameters(self, recurse: bool = True):
        """Return model parameters."""
        return self.model.parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Return named parameters."""
        return self.model.named_parameters(prefix=prefix, recurse=recurse)
    
    def get_adapter_parameters(self, agent_id: int):
        """
        Get parameters for a specific agent's adapter.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            Iterator of (name, parameter) tuples for the adapter
        """
        adapter_name = f"agent_{agent_id}"
        for name, param in self.model.named_parameters():
            if adapter_name in name and param.requires_grad:
                yield name, param
    
    def get_all_adapter_parameters(self) -> Dict[int, List[Tuple[str, torch.nn.Parameter]]]:
        """
        Get parameters for all adapters.
        
        Returns:
            Dict mapping agent_id to list of (name, parameter) tuples
        """
        result = {i: [] for i in range(self.num_agents)}
        
        for name, param in self.model.named_parameters():
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
        
        for name, param in self.model.named_parameters():
            if "lora" in name.lower():
                if active_adapter in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    
    def unfreeze_all_adapters(self) -> None:
        """Unfreeze all adapter parameters."""
        for name, param in self.model.named_parameters():
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
        self.model.save_pretrained(
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
        self.model.load_adapter(load_path, adapter_name=adapter_name)
        logger.info(f"Loaded adapter {adapter_name} from {load_path}")
    
    @property
    def config(self):
        """Return the base model's config."""
        return self.model.config
    
    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing."""
        self.model.gradient_checkpointing_enable(**kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.model.gradient_checkpointing_disable()


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

