# Multi-Agent LoRA Adapter Framework - Progress Tracker

## Project Overview

Implementing an RL finetuning framework with separate LoRA adapters per agent.
Each adapter only updates during backprop for its respective agent's trajectories.

## Status: TRAINING ACTIVELY RUNNING

### Current Phase
- [x] Phase 0: Environment Setup (COMPLETE)
- [x] Phase 1: Configuration and Data Flow (COMPLETE)
- [x] Phase 2: Multi-Adapter Model Implementation (COMPLETE)
- [x] Phase 3: Training Loop Modification (COMPLETE)
- [ ] Phase 4: Inference Adaptation (Optional - future work)
- [x] Phase 5: Testing and Validation (COMPLETE)
- [x] Phase 6: Full Multi-Agent Training with WandB (RUNNING)

### Full Training Command
```bash
cd /workspace/MARS
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate mars-multi-agent

export ROLL_OUTPUT_DIR=/workspace/MARS/output/multi_agent_neworg
mkdir -p $ROLL_OUTPUT_DIR

python examples/start_multi_agent_pipeline.py \
    --config_path examples/hanabi \
    --config_name multi_agent_hanabi_wandb \
    --num_agents 2 \
    --lora_rank 32 \
    --lora_alpha 32
```

### Active Training Run
- WandB: https://wandb.ai/mars-hanabi/mars-multi-agent-hanabi/runs/bjty2azh
- Output: `/workspace/MARS/output/multi_agent_neworg`
- Log: `/tmp/training_neworg.log`

---

## Environment Information

### Hardware
- GPU: 2x NVIDIA A100-SXM4-80GB
- CUDA Version: 13.0 (system) / 12.4 (PyTorch bundled)
- PID Limit: 4194304 (sufficient for Ray)

### Conda Environment
- Name: `mars-multi-agent`
- Location: `/venv/mars-multi-agent`
- Activation: `source /opt/miniforge3/etc/profile.d/conda.sh && conda activate mars-multi-agent`

### Installed Versions
| Package | Version | Status |
|---------|---------|--------|
| Python | 3.10.19 | OK |
| torch | 2.6.0+cu124 | OK |
| vllm | 0.8.4 | OK |
| peft | 0.12.0 | OK |
| transformers | 4.57.3 | OK |
| deepspeed | 0.18.3 | OK |
| ray | 2.53.0 | OK |
| open_spiel | 1.6.10 | OK |
| wandb | 0.23.1 | OK |

Note: Using SDPA attention (flash-attn skipped due to CUDA version mismatch).

---

## Changelog

### 2025-12-29: Extended Training Run (Latest)
- **Active Training Session**
  - Pipeline progressing through steps 7, 8, 9+ continuously
  - Train rollouts completing: 64/64 trajectories per step
  - Per-agent training: agent_0 completing 8/8 gradient updates per step (~5.5s per iteration)
  - Weight updates: 758/758 parameters syncing successfully
  - 199 metrics logged to WandB per step
  
- **WandB Configuration Updated**
  - Organization changed to `mars-hanabi`
  - Project: `mars-multi-agent-hanabi`
  - Active run: https://wandb.ai/mars-hanabi/mars-multi-agent-hanabi/runs/bjty2azh

- **Observed Training Behavior**
  - Episodes starting with score -10.0 (expected for untrained model)
  - Model generating `<think>` reasoning tokens correctly
  - Memory efficiently managed: ~21.5GB per GPU, sleep mode freeing 36GB when not in use
  - vLLM prefix cache resetting between steps

### 2025-12-29: Full Training Working
- **Multi-Agent LoRA Training Successfully Running**
  - Created `MultiAgentDeepSpeedStrategy` to apply PEFT before DeepSpeed initialization
  - Fixed PEFT parameter name mapping for vLLM compatibility (strips `base_model.model.` and `.base_layer.` prefixes)
  - Both validation and training rollouts completing successfully
  - WandB logging active at `mars-multi-agent-hanabi` project
  
- **Key Architecture Changes**:
  - `roll/multi_agent/strategy.py`: New DeepSpeed strategy that injects PEFT adapters before DeepSpeed wrapping
  - `roll/multi_agent/worker.py`: Modified to pass multi-agent config to strategy before initialization
  - `roll/multi_agent/pipeline.py`: Overrides parent init to ensure proper multi-agent config flow
  - `roll/distributed/strategy/factory.py`: Registered `multi_agent_deepspeed_train` strategy

- **Bug Fixes Applied**:
  - `roll/multi_agent/worker.py`: Fixed `DataProto.select()` to `DataProto.select_idxs()` API call
  - `roll/utils/functionals.py`: Fixed `masked_var()` to handle single-sample edge case (return 0 variance instead of error)

- **Testing Results**:
  - Using Qwen3-0.6B for faster iteration
  - `val rollout progress: 100%` 96/96
  - `train rollout progress: 100%` 64/64
  - Weight updates completing without errors
  - Multi-agent train step completed: `agent_0 step 0: 100%`
  - Pipeline progressing to step 1+

### 2025-12-29: Initial Training Setup
- Recreated conda environment `mars-multi-agent` on new container with 2x A100-80GB
- Installed all dependencies: PyTorch 2.6.0+cu124, vLLM 0.8.4, DeepSpeed 0.18.3
- Pre-downloaded Qwen/Qwen3-4B model to avoid concurrent download issues
- Created `examples/hanabi/multi_agent_hanabi_wandb.yaml` with:
  - WandB tracking enabled (project: mars-multi-agent-hanabi)
  - Full training params from original `agentic_val_hanabi_selfplay.yaml`
  - Qwen3-4B model, 200 training steps
  - Per-agent LoRA adapters (rank 32, alpha 32)
- Container has sufficient PID limits for full Ray pipeline

### 2025-12-26: Integration Testing Complete
- Fixed Ray 2.53+ LogMonitor API compatibility (`roll/distributed/scheduler/log_monitor.py`)
- Fixed DeepSpeed+Triton+Ray issue by upgrading DeepSpeed to 0.18.3
- Updated config to use Qwen3-0.6B model for resource-constrained testing
- Added `RAY_NUM_CPUS` support in initialize.py to limit Ray worker spawning
- Verified pipeline initialization works with Hanabi self-play config
- Workers start correctly: actor_train, reference, EnvironmentWorker
- Note: Container thread limits may require running on dedicated hardware

### 2025-12-26: Unit Tests Passed
- Created test suite in `tests/multi_agent/`:
  - `test_gradient_isolation.py`: Verifies gradient isolation between adapters
  - `test_training_flow.py`: Tests batch partitioning and sequential training
- All tests passing:
  - Agent ID extraction from group_ids (e.g., "env_0_p1" -> agent_id=1)
  - Adapter switching (set_active_agent)
  - Gradient isolation (agent 0 gradients don't affect agent 1 and vice versa)
  - Separate adapter outputs (after training agent 0, only agent 0's output changes)
  - Batch partitioning by agent_id
  - Sequential training with per-agent metrics

### 2025-12-26: Multi-Agent Module Implementation Complete
- Created `roll/multi_agent/` module with core components:
  - `config.py`: MultiAgentConfig and MultiAgentLoRAConfig dataclasses
  - `model.py`: MultiAgentLoRAModel wrapper for managing multiple adapters
  - `worker.py`: MultiAgentActorWorker for per-agent training
  - `pipeline.py`: MultiAgentAgenticPipeline integration
- Created `examples/start_multi_agent_pipeline.py` launch script
- Key features implemented:
  - PEFT-based multi-adapter management using `add_adapter()` and `set_adapter()`
  - Agent ID extraction from group_ids (e.g., "env_0_p1" -> agent_id=1)
  - Sequential per-agent training strategy with gradient isolation
  - Batch partitioning by agent_id
  - Per-agent metrics tracking (score/mean, score/max, num_samples)
  - Separate adapter checkpointing
- All imports verified working

### 2025-12-26: Environment Setup Complete
- Created conda environment `mars-multi-agent` with Python 3.10
- Installed PyTorch 2.6.0+cu124, vLLM 0.8.4, PEFT 0.12.0
- Verified CUDA access with 2x A100 GPUs
- All MARS framework imports working correctly
- Note: Using SDPA attention (flash-attn skipped due to CUDA version mismatch)

### 2025-12-26: Project Initialization
- Created Progress.md
- Analyzed existing MARS/ROLL codebase
- Drafted engineering plan for multi-agent LoRA system
- Key insight: PEFT supports multiple named adapters via `add_adapter()` and `set_adapter()`
- Recommended implementation: Sequential per-adapter training for simplicity

---

## Architecture Design

### Core Components

1. **MultiAgentLoRAModel**: Wrapper holding N LoRA adapters
2. **Agent-Trajectory Routing**: Tags samples with agent_id, routes to correct adapter
3. **Gradient Isolation**: Per-agent backward passes with adapter switching

### Key Design Decisions

1. **Sequential Training**: Train one adapter at a time per batch partition
   - Rationale: Simpler, debuggable, DeepSpeed compatible
   
2. **PEFT Multi-Adapter**: Leverage existing PEFT infrastructure
   - Use `model.add_adapter(adapter_name, config)`
   - Switch via `model.set_adapter(adapter_name)`

3. **Agent ID Propagation**: Add `agent_id` to DataProto non_tensor_batch
   - Already have `player_id` in EnvManager, propagate to training

---

## Implementation Notes

### Git Commit Strategy
Commits will be batched logically:
1. Environment setup commit
2. Configuration changes commit
3. Multi-adapter model commit
4. Training loop changes commit
5. Inference changes commit
6. Testing commit

(Commits will be provided as messages for user to execute)

---

## Reproduction Instructions

### Step 1: Environment Setup
```bash
# Initialize conda
source /opt/miniforge3/etc/profile.d/conda.sh

# Create conda environment
conda create -n mars-multi-agent python=3.10 -y
conda activate mars-multi-agent

# Install PyTorch with CUDA
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install vLLM (this may upgrade torch)
pip install vllm==0.8.4

# Install other dependencies
pip install numpy"<2.0a0,>=1.25" ray tensordict transformers==4.51.2 \
    datasets==3.1.0 peft==0.12.0 accelerate==0.34.2 deepspeed==0.16.0 \
    trl==0.9.6 hydra-core omegaconf wandb matplotlib imageio open-spiel

# Install mcore_adapter from local
cd /workspace/MARS
pip install ./mcore_adapter
```

### Step 2: Verify GPU Access
```bash
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate mars-multi-agent
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

### Step 3: Verify MARS Imports
```bash
cd /workspace/MARS
python -c "
import sys; sys.path.insert(0, '.')
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline
print('All imports successful!')
"
```

### Step 4: Run Unit Tests
```bash
cd /workspace/MARS
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate mars-multi-agent

# Run gradient isolation tests
python tests/multi_agent/test_gradient_isolation.py

# Run training flow tests
python tests/multi_agent/test_training_flow.py
```

### Step 5: Run Multi-Agent Pipeline (Hanabi)
```bash
cd /workspace/MARS
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate mars-multi-agent

# Set environment variables
export ROLL_OUTPUT_DIR=/workspace/MARS/output/hanabi_multi_agent_$(date +%Y%m%d-%H%M%S)
export ROLL_LOG_DIR=$ROLL_OUTPUT_DIR/logs
mkdir -p $ROLL_LOG_DIR

# Run multi-agent pipeline
python examples/start_multi_agent_pipeline.py \
    --config_path hanabi \
    --config_name agentic_val_hanabi_selfplay \
    --num_agents 2 \
    --lora_rank 32 \
    --lora_alpha 32
```

---

## File Changes Tracking

| File | Status | Description |
|------|--------|-------------|
| roll/multi_agent/__init__.py | DONE | Module initialization and exports |
| roll/multi_agent/config.py | DONE | MultiAgentConfig and MultiAgentLoRAConfig |
| roll/multi_agent/model.py | DONE | MultiAgentLoRAModel wrapper with PEFT |
| roll/multi_agent/worker.py | DONE | MultiAgentActorWorker for per-agent training |
| roll/multi_agent/pipeline.py | DONE | MultiAgentAgenticPipeline integration |
| examples/start_multi_agent_pipeline.py | DONE | Launch script for multi-agent training |
| tests/multi_agent/test_gradient_isolation.py | DONE | Unit tests for gradient isolation |
| tests/multi_agent/test_training_flow.py | DONE | Unit tests for training flow |
| .gitignore | DONE | Added wandb/ and output/ |
| roll/distributed/scheduler/log_monitor.py | DONE | Ray 2.53+ compatibility fix |
| roll/distributed/scheduler/initialize.py | DONE | Added RAY_LOCAL_MODE support |
| examples/hanabi/multi_agent_hanabi_test.yaml | DONE | Test config for Hanabi |
| examples/run_multi_agent_no_ray.py | DONE | Non-Ray training script for containers |
| tests/multi_agent/test_local_training.py | DONE | Local training test (no Ray) |
| examples/hanabi/multi_agent_hanabi_wandb.yaml | DONE | Full training config with WandB |

---

## Git Commit Messages

Use these commit messages for the changes (in order):

### Commit 1: Multi-Agent LoRA Module
```
feat: Add multi-agent LoRA adapter framework

- Add roll/multi_agent/ module with core components:
  - config.py: MultiAgentConfig and MultiAgentLoRAConfig dataclasses
  - model.py: MultiAgentLoRAModel wrapper for PEFT multi-adapter
  - worker.py: MultiAgentActorWorker with per-agent training
  - pipeline.py: MultiAgentAgenticPipeline integration
- Implement gradient isolation via adapter freezing
- Support agent_id extraction from group_ids (e.g., "env_0_p1" -> agent_id=1)
- Add per-agent metrics collection
```

### Commit 2: Test Suite
```
test: Add unit tests for multi-agent LoRA framework

- Add tests/multi_agent/test_gradient_isolation.py
  - Test adapter switching
  - Verify gradient isolation between agents
  - Test separate adapter outputs
- Add tests/multi_agent/test_training_flow.py
  - Test batch partitioning by agent_id
  - Test sequential training with gradient isolation
  - Test per-agent metrics collection
```

### Commit 3: Examples and Configuration
```
feat: Add multi-agent pipeline launcher and Hanabi config

- Add examples/start_multi_agent_pipeline.py launcher
- Add examples/hanabi/multi_agent_hanabi_test.yaml test config
- Update .gitignore to ignore wandb/ and output/ directories
```

---

## Next Steps

1. **Monitor Active Training**: Training is running at https://wandb.ai/mars-hanabi/mars-multi-agent-hanabi/runs/bjty2azh
   - Track `critic/score/mean` for improvement over episodes
   - Monitor per-agent metrics: `train/agent_0/score/mean`

2. **Scale to Larger Model**: Once training loop validated, switch to Qwen3-4B for production runs

3. **VLLM Adapter Switching (Optional)**: Modify inference to switch adapters per-agent during generation

4. **Multi-Agent Differentiation**: Currently training only agent_0; extend to train both agents with separate adapters

---

## References

- PEFT Documentation: https://huggingface.co/docs/peft
- VLLM LoRA: https://docs.vllm.ai/en/latest/models/lora.html
- DeepSpeed: https://www.deepspeed.ai/
