"""
Multi-Agent LoRA Adapter Framework

This module provides support for training multiple LoRA adapters,
one per agent, with gradient isolation during backpropagation.

Key Components:
- MultiAgentConfig: Configuration for multi-agent training
- MultiAgentLoRAModel: Model wrapper that manages multiple LoRA adapters
- MultiAgentActorWorker: Modified actor worker for per-agent training
"""

from roll.multi_agent.config import MultiAgentConfig, MultiAgentLoRAConfig
from roll.multi_agent.model import MultiAgentLoRAModel, create_multi_agent_model
from roll.multi_agent.worker import MultiAgentActorWorker, get_multi_agent_worker_class
from roll.multi_agent.pipeline import MultiAgentAgenticPipeline, create_multi_agent_pipeline

__all__ = [
    "MultiAgentConfig",
    "MultiAgentLoRAConfig",
    "MultiAgentLoRAModel",
    "create_multi_agent_model",
    "MultiAgentActorWorker",
    "get_multi_agent_worker_class",
    "MultiAgentAgenticPipeline",
    "create_multi_agent_pipeline",
]

