# MARS Project Progress

## Current Experiments

### Multi-Agent LoRA Adapter Comparison Study

**Goal**: Compare MARS vs PubMDP training approaches using separate LoRA adapters for each agent.

| Config | Approach | Train Environment | Eval Environments | Status |
|--------|----------|-------------------|-------------------|--------|
| `mars_train_mini_hanabi.yaml` | MARS (separate reward norm) | MiniHanabi | MiniHanabi, SimpleHanabi | Running on GPU 0 |
| `pubmdp_train_mini_hanabi.yaml` | PubMDP (shared reward norm) | MiniHanabi | MiniHanabi, SimpleHanabi | Pending |

**Key Difference**:
- MARS: `separate_norm_for_selfplay: true` - Each agent's rewards normalized separately
- PubMDP: `separate_norm_for_selfplay: false` - Shared reward normalization across agents

**Branch**: `koh-dev/ma-training`

---

### Bridge Game Validation (Sanity Check)

**Goal**: Validate bridge game training code on a smaller model before full-scale experiments.

| Config | Model | Environment | Status |
|--------|-------|-------------|--------|
| `agentic_val_tiny_bridge_selfplay.yaml` | Qwen3-4B | TinyBridge | Completed 2 steps, intermittent crash |

**Branch**: `koh-dev/bridges-validation`

**Results** (2026-01-09):
- Step 0: Completed successfully
  - `env/TinyBridge/success`: 0.0
  - `critic/score/mean`: -51.23
  - `system/tps`: 47.46 tokens/sec
  - Model reasoning coherently (bidding 2NT with strong hands)
- Step 1: Completed successfully
  - `critic/score/mean`: -48.70
  - `system/tps`: 48.59 tokens/sec
- Step 2: Crashed at 66% rollout with `TypeError: list indices must be integers or slices, not NoneType`

**Issue**: Intermittent crash during rollout, likely related to `next_player` being `None` when game terminates.

---

## Infrastructure Notes

- **Server**: odin2
- **Cache Directory**: `/ext_hdd2/nhkoh`
- **Conda Environment**: `mars-multi-agent`

### GPU Allocation Challenges

Ray's internal GPU management overrides `CUDA_VISIBLE_DEVICES` for workers, making it difficult to run multiple Ray-based training sessions on different GPUs simultaneously. Current workaround: sequential execution.

---

## Session Log

### 2026-01-08

- MARS training initiated on MiniHanabi (GPU 0)
- Encountered and resolved CUDA OOM issues by reducing batch sizes
- Attempted parallel PubMDP training on GPU 2 - failed due to Ray GPU allocation
- Switched to `koh-dev/bridges-validation` branch for bridge code validation

### 2026-01-09

- Attempted Qwen3-0.6B validation - model too small, generated garbage
- Switched to Qwen3-4B for bridge validation
- Resolved vLLM BFloat16 serialization issues with `VLLM_USE_V1=0`
- Completed 2 training steps successfully before intermittent crash
- Model shows coherent reasoning in Tiny Bridge game
- Identified potential bug: `next_player` set to `None` when game terminates may cause downstream issues




