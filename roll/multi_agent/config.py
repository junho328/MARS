"""
Multi-Agent LoRA Configuration

Defines configuration classes for multi-agent LoRA training where
each agent has its own dedicated LoRA adapter.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class MultiAgentLoRAConfig:
    """Configuration for a single agent's LoRA adapter."""
    
    adapter_name: str = field(
        default="default",
        metadata={"help": "Name identifier for this adapter (e.g., 'agent_0', 'agent_1')"}
    )
    
    lora_rank: int = field(
        default=32,
        metadata={"help": "Rank of the LoRA decomposition"}
    )
    
    lora_alpha: int = field(
        default=32,
        metadata={"help": "Alpha parameter for LoRA scaling"}
    )
    
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout probability for LoRA layers"}
    )
    
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of module names to apply LoRA to. None uses default."}
    )
    
    def to_peft_config(self):
        """Convert to PEFT LoraConfig."""
        from peft import LoraConfig
        
        target = self.target_modules
        if target is None:
            target = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        return LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target,
            bias="none",
            task_type="CAUSAL_LM",
        )


@dataclass
class MultiAgentConfig:
    """
    Configuration for multi-agent LoRA training.
    
    Enables training with separate LoRA adapters per agent,
    where gradients only flow through the corresponding agent's adapter.
    """
    
    enabled: bool = field(
        default=False,
        metadata={"help": "Whether multi-agent LoRA training is enabled"}
    )
    
    num_agents: int = field(
        default=2,
        metadata={"help": "Number of agents (and LoRA adapters) to create"}
    )
    
    shared_base_model: bool = field(
        default=True,
        metadata={"help": "Whether agents share the same base model weights"}
    )
    
    adapter_configs: Optional[Dict[str, MultiAgentLoRAConfig]] = field(
        default=None,
        metadata={"help": "Per-agent LoRA configurations. If None, uses default for all."}
    )
    
    default_lora_rank: int = field(
        default=32,
        metadata={"help": "Default LoRA rank for all adapters"}
    )
    
    default_lora_alpha: int = field(
        default=32,
        metadata={"help": "Default LoRA alpha for all adapters"}
    )
    
    default_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )
    
    training_strategy: str = field(
        default="sequential",
        metadata={
            "help": "Training strategy: 'sequential' (train one adapter at a time) "
                    "or 'parallel' (custom gradient routing)"
        }
    )
    
    def __post_init__(self):
        """Initialize adapter configs if not provided."""
        if self.enabled and self.adapter_configs is None:
            self.adapter_configs = {}
            target_modules = [m.strip() for m in self.default_target_modules.split(",")]
            
            for i in range(self.num_agents):
                adapter_name = f"agent_{i}"
                self.adapter_configs[adapter_name] = MultiAgentLoRAConfig(
                    adapter_name=adapter_name,
                    lora_rank=self.default_lora_rank,
                    lora_alpha=self.default_lora_alpha,
                    target_modules=target_modules,
                )
    
    def get_adapter_names(self) -> List[str]:
        """Get list of all adapter names."""
        if not self.enabled:
            return []
        return [f"agent_{i}" for i in range(self.num_agents)]
    
    def get_adapter_config(self, agent_id: int) -> MultiAgentLoRAConfig:
        """Get LoRA config for a specific agent."""
        adapter_name = f"agent_{agent_id}"
        if self.adapter_configs and adapter_name in self.adapter_configs:
            return self.adapter_configs[adapter_name]
        
        target_modules = [m.strip() for m in self.default_target_modules.split(",")]
        return MultiAgentLoRAConfig(
            adapter_name=adapter_name,
            lora_rank=self.default_lora_rank,
            lora_alpha=self.default_lora_alpha,
            target_modules=target_modules,
        )
    
    def agent_id_from_group_id(self, group_id: str) -> int:
        """
        Extract agent_id from a group_id string.
        
        Group IDs in self-play mode have format: "{base_id}_p{player_id}"
        For example: "env_0_p1" -> agent_id = 1
        """
        if "_p" in str(group_id):
            parts = str(group_id).rsplit("_p", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return int(parts[1])
        return 0

