# Multi-Agent LoRA Adapter Framework - Progress

## Overview
Implementation of a multi-agent RL finetuning framework where each agent has its own dedicated LoRA adapter. The framework supports training on Hanabi and Bridge environments.

## Architecture
```
+-------------------+
|   Base Model      |  (frozen weights, shared across all agents)
+-------------------+
         |
    +----+----+
    |         |
+---v---+ +---v---+
|Adapter| |Adapter|
|  A0   | |  A1   |
+-------+ +-------+
   |         |
Agent 0   Agent 1
```

## Implementation Status

### Phase 1: Core Infrastructure - COMPLETED
- [x] `roll/models/multi_adapter_manager.py` - Multi-adapter management class
- [x] `roll/configs/model_args.py` - Added LoRA config fields
- [x] `roll/pipeline/agentic/agentic_config.py` - Added multi-adapter config

### Phase 2: Model Provider Integration - COMPLETED
- [x] `roll/models/model_providers.py` - Multi-adapter model loading

### Phase 3: Training Pipeline Modifications - COMPLETED
- [x] `roll/pipeline/base_worker.py` - Training with per-agent adapters
- [x] `roll/pipeline/agentic/agentic_pipeline.py` - Pipeline modifications
- [x] Loss function updates for gradient isolation

### Phase 4: Inference Integration - COMPLETED
- [x] `roll/agentic/rollout/env_manager.py` - Added agent_ids to rollout data

### Phase 5: DeepSpeed Strategy Updates - COMPLETED
- [x] `roll/distributed/strategy/deepspeed_strategy.py` - Multi-adapter support

### Phase 6: Configuration and Examples - COMPLETED
- [x] `examples/hanabi/multi_agent_hanabi.yaml` - Hanabi multi-agent config

## Environment Setup

### Prerequisites
```bash
# Create conda environment
conda create -n mars-multi-adapter python=3.10 -y
conda activate mars-multi-adapter

# Install PyTorch (adjust for your CUDA version)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install MARS dependencies
cd /workspace/MARS
pip install -e .
pip install -r requirements_common.txt
```

### Key Dependencies
- peft==0.12.0 (for LoRA adapters)
- transformers==4.51.2
- deepspeed
- ray

## Usage

### Configuration
Add these settings to your YAML config:

```yaml
# Enable multi-adapter mode
multi_adapter_mode: true
num_agents: 2  # Number of agents (each gets a LoRA adapter)

# LoRA configuration
lora_target: q_proj,k_proj,v_proj,o_proj
lora_rank: 32
lora_alpha: 32

# In actor_train.model_args:
actor_train:
  model_args:
    multi_adapter_mode: true
    num_agents: 2
    lora_target: ${lora_target}
    lora_rank: ${lora_rank}
    lora_alpha: ${lora_alpha}
    freeze_base_model: true
```

### Running Multi-Agent Hanabi Training
```bash
python examples/start_agentic_pipeline.py --config examples/hanabi/multi_agent_hanabi.yaml
```

## Key Components

### MultiAdapterManager (`roll/models/multi_adapter_manager.py`)
Central class for managing multiple LoRA adapters:
- `create_adapters(num_agents, lora_config)`: Create N adapters
- `activate_adapter(agent_id)`: Switch active adapter
- `get_adapter_name(agent_id)`: Return adapter name for agent
- `save_adapters(path)` / `load_adapters(path)`: Persistence

### Gradient Isolation
Each adapter only receives gradients from its own agent's data:
1. Data is grouped by `agent_id` in `_train_step_multi_adapter`
2. For each agent group, the corresponding adapter is activated
3. Gradients only flow to the active adapter (others have `requires_grad=False`)

### Agent ID Tracking
Agent IDs are tracked through the rollout pipeline:
1. `env_manager.py` adds `agent_ids` to rollout data based on player_id
2. Group IDs include player suffix: `group_id_p0`, `group_id_p1`
3. Pipeline extracts agent_ids via `_extract_agent_ids()`

## Git Commits (Suggested)
After reviewing the changes, commit with:

```bash
git add roll/models/multi_adapter_manager.py
git add roll/configs/model_args.py
git add roll/pipeline/agentic/agentic_config.py
git add roll/models/model_providers.py
git add roll/pipeline/base_worker.py
git add roll/pipeline/agentic/agentic_pipeline.py
git add roll/agentic/rollout/env_manager.py
git add roll/distributed/strategy/deepspeed_strategy.py
git add examples/hanabi/multi_agent_hanabi.yaml
git add Progress.md

git commit -m "feat: add multi-agent LoRA adapter framework for per-agent training

- Add MultiAdapterManager for managing multiple LoRA adapters
- Each agent gets a dedicated adapter (adapter_0, adapter_1, etc.)
- Gradient isolation ensures each adapter only learns from its agent
- Support for Hanabi and Bridge self-play training
- Add multi_adapter_mode config option
- Add example config for multi-agent Hanabi training"
```

## Testing

### Unit Test for MultiAdapterManager
```python
from roll.models.multi_adapter_manager import MultiAdapterManager, AdapterConfig
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.5B")

# Create manager with 2 adapters
config = AdapterConfig(lora_rank=16, lora_alpha=16)
manager = MultiAdapterManager(model, num_agents=2, adapter_config=config)

# Test adapter switching
manager.activate_adapter(0)
assert manager.active_adapter == "adapter_0"

manager.activate_adapter(1)
assert manager.active_adapter == "adapter_1"

# Test gradient isolation
manager.activate_adapter(0, enable_gradient=True)
for name, param in manager.model.named_parameters():
    if "lora_" in name:
        if "adapter_0" in name:
            assert param.requires_grad == True
        else:
            assert param.requires_grad == False
```

## Notes
- Hanabi: 2-player cooperative game
- Bridge: 4-player game (can be configured as 2 teams with 2 adapters)
- Each adapter is independent during backprop
- Base model weights remain frozen
- Adapters are saved separately for independent loading

## Collaborator Instructions
1. Ensure conda environment is activated: `conda activate mars-multi-adapter`
2. Check implementation status above
3. Run the example config to verify: `examples/hanabi/multi_agent_hanabi.yaml`
4. See individual file docstrings for API details
