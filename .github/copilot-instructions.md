# MARSHAL Copilot Instructions

## Project Overview
MARSHAL is a reinforcement learning framework for multi-agent reasoning through self-play with strategic LLMs, built on top of the [ROLL framework](https://github.com/alibaba/ROLL). The project trains agents on competitive and cooperative games (TicTacToe, Connect Four, Hanabi, Poker variants) to improve reasoning capabilities that generalize to benchmarks like AIME and GPQA-Diamond.

## Architecture

### Core Components
- **`roll/pipeline/`**: Pipeline orchestration layer
  - `agentic/`: Multi-agent agentic RL pipeline (MARSHAL's main pipeline)
  - `rlvr/`: RLVR (Reinforcement Learning from Value Refinement) pipeline
  - `base_pipeline.py`: Common pipeline interface with checkpoint management, model updates, and tracking

- **`roll/agentic/`**: Agentic RL implementation
  - `env/`: Game environments (`tictactoe/`, `connect_four/`, `hanabi/`, `kuhn_poker/`, `leduc_poker/`)
  - `rollout/`: Rollout scheduling and environment management
    - `env_manager.py`: Multi-threaded environment manager (handles player trajectories, action parsing, reward computation)
    - `rollout_scheduler.py`: Coordinates rollouts across distributed workers

- **`roll/distributed/`**: Ray-based distributed execution
  - `executor/cluster.py`: Ray cluster management for actor/critic/reference models
  - `scheduler/`: Request scheduling and resource management

- **`roll/models/`**: Model providers (wraps transformers, vLLM, SGLang)

### Key Abstractions
- **Cluster**: Ray-based distributed worker group for model replicas (actor_train, actor_infer, reference, critic)
- **EnvManager**: Manages parallel game environments, handles multi-turn trajectories with player roles
- **BaseEnv**: Abstract environment interface (`reset()`, `step()`, `render()`)
  - All game envs in `roll/agentic/env/` must register in `REGISTERED_ENVS` dict

## Configuration System

### Hydra-based YAML Configs
Configurations use Hydra composition with defaults from `examples/config/`:
- `envs@_here_`: Environment definitions
- `deepspeed_zero*@_here_`: DeepSpeed training strategies

Example config structure ([examples/tictactoe/agentic_val_tictactoe_selfplay.yaml](examples/tictactoe/agentic_val_tictactoe_selfplay.yaml)):
```yaml
defaults:
  - ../config/envs@_here_
  - ../config/deepspeed_zero@_here_

pretrain: /path/to/model  # Base model checkpoint
max_steps: 200
rollout_batch_size: 128
train_env_manager:
  env_groups: 128
  tags: ["tictactoe"]
```

### Config Dataclasses
All configs inherit from `@dataclass` types:
- `AgenticConfig` (pipeline-level): Defines actor/critic clusters, env managers, training hyperparams
- `EnvManagerConfig`: Environment parallelism settings (`env_groups`, `group_size`, `max_env_num_per_worker`)
- `WorkerConfig`: Base for all worker configurations (actor_train, actor_infer, reference, critic)

## Development Workflows

### Running Training
Always use bash scripts in `examples/<game>/`:
```bash
# Self-play training
bash examples/tictactoe/run_agentic_pipeline_tictactoe_selfplay.sh

# Training outputs to: ./runs/<game>_<mode>/<timestamp>/
export ROLL_OUTPUT_DIR=./runs/...  # Set by scripts
export ROLL_LOG_DIR=$ROLL_OUTPUT_DIR/logs
export ROLL_RENDER_DIR=$ROLL_OUTPUT_DIR/render
```

Scripts execute: `python examples/start_agentic_pipeline.py --config_path tictactoe --config_name agentic_val_tictactoe_selfplay`

### Monitoring
```bash
# TensorBoard tracking (default)
tensorboard --logdir=runs/tictactoe_selfplay/

# Logs stored in ROLL_LOG_DIR/custom_logs.log
```

### Testing
```bash
make test  # Run pytest with auto parallelization
```

### Model Conversion
```bash
bash model_convert.sh  # Export checkpoints for evaluation
```

## Project-Specific Conventions

### Environment Registration
To add new environments:
1. Create directory in `roll/agentic/env/<env_name>/`
2. Implement `BaseEnv` or `BaseLanguageBasedEnv` in `env.py`
3. Create `@dataclass` config inheriting `BaseEnvConfig` in `config.py`
4. Register in `roll/agentic/env/__init__.py`:
   ```python
   REGISTERED_ENVS["my_env"] = MyEnv
   REGISTERED_ENV_CONFIGS["my_env"] = MyEnvConfig
   ```

### Multi-Agent Self-Play
- Environments track player-specific rewards via `player_id` in trajectories
- `use_turn_scores=True`: Enables turn-level advantage estimation (key to MARSHAL)
- `reward_normalization.separate_norm_for_selfplay=True`: Normalizes rewards per player

### Special Tokens
MARSHAL uses think/answer delimiters for reasoning traces:
```python
special_token_list: ["<think>", "</think>", "<answer>", "</answer>"]
```
Set `enable_think=False` to disable reasoning token RL.

### Ray Cluster Management
- Framework automatically manages Ray clusters (don't call `ray.init()` manually)
- Scripts call `ray stop` before execution
- Namespace controlled via `RAY_NAMESPACE` in `roll/utils/constants.py`

## Code Style
- **Formatting**: Black (line length 119), Ruff linting
- **Imports**: `known-first-party = ["roll"]` in ruff config
- **Type hints**: Extensive use of typing annotations
- **Logging**: Use `from roll.utils.logging import get_logger` (never print statements)

## Common Pitfalls
- **Don't modify YAML configs directly in repo**: Copy to custom location and edit
- **Worker world_size auto-calculated**: Set `max_env_num_per_worker` in `EnvManagerConfig`, not `world_size` directly
- **Model paths**: Config examples use placeholder paths (`/mnt/public/...`); update `pretrain` field to your model location
- **Checkpoint loading**: Use `resume_from_checkpoint` in config, not ad-hoc loading logic

## Key Files for Understanding
- [roll/pipeline/agentic/agentic_pipeline.py](roll/pipeline/agentic/agentic_pipeline.py): Main training loop
- [roll/agentic/rollout/env_manager.py](roll/agentic/rollout/env_manager.py): Multi-agent trajectory management
- [roll/agentic/env/base.py](roll/agentic/env/base.py): Environment interface contracts
- [examples/start_agentic_pipeline.py](examples/start_agentic_pipeline.py): Entry point using Hydra configs
