# MARS-LLM Codebase Guide

## Overview
MARS (Multi-Agent Reinforcing Self-play) is a framework for training large language models through reinforcement learning in strategic games. This guide provides a comprehensive walkthrough of the codebase using **Hanabi** as the primary example.

---

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Hanabi Example Walkthrough](#hanabi-example-walkthrough)
5. [Key Files and Their Roles](#key-files-and-their-roles)
6. [Pipeline Flow](#pipeline-flow)
7. [Configuration System](#configuration-system)
8. [Environment System](#environment-system)
9. [Model Components](#model-components)
10. [Running Examples](#running-examples)

---

## Environment Setup

This section provides detailed instructions for setting up the MARS-LLM environment using Python virtual environments and `uv` (a fast Python package installer).

### Prerequisites

Before starting, ensure you have:
- **Python 3.10 or 3.11** installed (check with `python --version`)
- **CUDA 11.8 or 12.1** for GPU support (check with `nvidia-smi`)
- **Git** for cloning the repository
- **Build tools** for compiling dependencies:
  ```bash
  # Ubuntu/Debian
  sudo apt-get update
  sudo apt-get install -y build-essential python3-dev git
  
  # Check Python version (should be 3.10 or 3.11)
  python3 --version
  ```

### Step 1: Clone the Repository

```bash
cd ~
git clone https://github.com/your-org/MARS-LLM.git
cd MARS-LLM
```

### Step 2: Create Python Virtual Environment

Using Python's built-in `venv` module (no Anaconda required):

**Option A: Use system default Python (if it's already 3.10 or 3.11)**
```bash
# Check your Python version first
python3 --version

# Create virtual environment in .venv directory
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Verify you're in the virtual environment
which python
# Should show: /home/yourusername/MARS-LLM/.venv/bin/python

# Check Python version in venv
python --version

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel
```

**Option B: Use specific Python version (e.g., Python 3.10 or 3.11)**

If you have multiple Python versions installed:

```bash
# List available Python versions
ls /usr/bin/python*

# Create venv with specific Python version (Python 3.10)
python3.10 -m venv .venv

# Or for Python 3.11
python3.11 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Verify correct Python version
python --version
# Should show: Python 3.10.x or Python 3.11.x

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel
```

**Option C: Install specific Python version first (if not available)**

If you don't have Python 3.10 or 3.11 installed:

```bash
# Ubuntu/Debian - Add deadsnakes PPA for different Python versions
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update

# Install Python 3.10 (recommended)
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev

# Or install Python 3.11
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

# Now create venv with the specific version
python3.10 -m venv .venv
# or
python3.11 -m venv .venv

# Activate and verify
source .venv/bin/activate
python --version

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel
```

**Note:** Always activate the virtual environment before working with MARS-LLM:
```bash
cd ~/MARS-LLM
source .venv/bin/activate
```

To deactivate when done:
```bash
deactivate
```

### Step 3: Install uv (Fast Package Installer)

`uv` is a high-performance Python package installer written in Rust - much faster than pip.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version

# If uv command not found, add to PATH
export PATH="$HOME/.cargo/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Choose Your PyTorch and Backend Configuration

MARS-LLM supports different PyTorch versions and inference backends. Choose one based on your setup:

**Option A: PyTorch 2.5.1 + vLLM (Recommended for most users)**
```bash
# Install PyTorch 2.5.1 with CUDA 12.1
uv pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.8.4 (required version for MARS)
uv pip install vllm==0.8.4

# Install other dependencies
uv pip install -r requirements_torch251_vllm.txt
```

**Option B: PyTorch 2.6.0 + vLLM**
```bash
# Install PyTorch 2.6.0 with CUDA 12.1
uv pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.8.4
uv pip install vllm==0.8.4

# Install other dependencies
uv pip install -r requirements_torch260_vllm.txt
```

**Option C: PyTorch 2.5.1 + SGLang**
```bash
# Install PyTorch 2.5.1 with CUDA 12.1
uv pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
uv pip install -r requirements_torch251_sglang.txt
```

### Step 5: Install Common Dependencies

All configurations need these common dependencies:

```bash
# Install common requirements
uv pip install -r requirements_common.txt

# This includes:
# - ray (distributed computing)
# - transformers (model loading)
# - wandb (experiment tracking)
# - hydra-core (configuration management)
# - And many more...
```

### Step 6: Install mcore_adapter (Megatron-Core Adapter)

The `mcore_adapter` package is required for distributed training:

```bash
# Install in editable mode
cd mcore_adapter
uv pip install -e .
cd ..

# Verify installation
python -c "import mcore_adapter; print('mcore_adapter installed successfully')"
```

### Step 6.5: Download the Base Model (Qwen3-4B)

MARS-LLM uses **Qwen3-4B** as the base model (as used in the paper).

**Option A: Auto-download during first run (Easiest)**
```bash
# Model will download automatically from HuggingFace when you run training
# Default cache location: ~/.cache/huggingface/hub/

# To change cache location:
export HF_HOME=/ext_hdd/dematsunaga/huggingface_cache
echo 'export HF_HOME=/ext_hdd/dematsunaga/huggingface_cache' >> ~/.bashrc
```

**Option B: Pre-download using provided script (Recommended)**
```bash
# Use the download script
bash scripts/download_model.sh Qwen/Qwen3-4B /ext_hdd/dematsunaga/models

# This downloads to: /ext_hdd/dematsunaga/models/Qwen3-4B/
```

**Option C: Manual download using huggingface-cli**
```bash
# Install huggingface-cli (if not already installed)
uv pip install -U huggingface_hub

# Download Qwen3-4B
huggingface-cli download Qwen/Qwen3-4B \
  --local-dir /ext_hdd/dematsunaga/models/Qwen3-4B \
  --local-dir-use-symlinks False

# Verify download
ls -lh /ext_hdd/dematsunaga/models/Qwen3-4B/
# Should show: config.json, model safetensors files, tokenizer files, etc.
```

**Option D: Manual download using git**
```bash
# Install git-lfs
sudo apt-get install git-lfs
git lfs install

# Clone the model repository
cd /ext_hdd/dematsunaga/models/
git clone https://huggingface.co/Qwen/Qwen3-4B

# Verify
du -sh Qwen3-4B
# Should be ~8-9 GB
```

**Update configs to use local model path:**
```yaml
# In your YAML configs (e.g., agentic_rollout_hanabi.yaml)
pretrain: /ext_hdd/dematsunaga/models/Qwen3-4B

actor_infer:
  model_args:
    model_name_or_path: /ext_hdd/dematsunaga/models/Qwen3-4B
```

### Step 7: Install Game Environments

For Hanabi and other game environments:

```bash
# Install OpenSpiel for game environments
uv pip install open_spiel

# Install Gymnasium for RL environments
uv pip install "gymnasium[toy-text]"

# Install Sokoban environment
uv pip install gym_sokoban
```

### Step 8: Verify Installation

Check that everything is installed correctly:

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Check vLLM
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"

# Check Ray
python -c "import ray; print(f'Ray: {ray.__version__}')"

# Check transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check wandb
python -c "import wandb; print(f'WandB: {wandb.__version__}')"

# Check mcore_adapter
python -c "import mcore_adapter; print('mcore_adapter: OK')"

# Check OpenSpiel
python -c "import pyspiel; print('OpenSpiel: OK')"
```

**Expected output:**
```
PyTorch: 2.5.1+cu121
CUDA available: True
CUDA version: 12.1
vLLM: 0.8.4
Ray: 2.51.0
Transformers: 4.51.2
WandB: 0.19.1
mcore_adapter: OK
OpenSpiel: OK
```

### Step 9: Configure WandB (Optional but Recommended)

MARS-LLM uses Weights & Biases for experiment tracking:

```bash
# Login to wandb
wandb login

# Or set API key manually
export WANDB_API_KEY=your_api_key_here
echo 'export WANDB_API_KEY=your_api_key_here' >> ~/.bashrc
```

### Step 10: Set Up Environment Variables

Create a convenient setup script:

```bash
cat > setup_env.sh << 'EOF'
#!/bin/bash
# MARS-LLM Environment Setup Script

# Activate virtual environment
source .venv/bin/activate

# Set Python path
export PYTHONPATH="$PWD:$PYTHONPATH"

# Set wandb directory (optional)
export WANDB_DIR="/ext_hdd/dematsunaga"

# Ray configuration
export RAY_ADDRESS=""  # Force local Ray cluster
export RAY_DEDUP_LOGS=0  # Show all logs

# GPU settings
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0, adjust as needed

# Disable torch distributed debug (reduces noise)
export NCCL_DEBUG=WARN

echo "MARS-LLM environment activated!"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
EOF

chmod +x setup_env.sh
```

Now you can activate everything with:
```bash
source setup_env.sh
```

### Step 11: Test Installation with Hanabi

Run a quick test to verify everything works:

```bash
# Activate environment
source setup_env.sh

# Run a small Hanabi rollout test
cd examples/hanabi

# Create a test config (already provided)
cat agentic_rollout_hanabi.yaml

# Run the test
bash run_agentic_rollout_hanabi.sh
```

If successful, you should see:
- Ray cluster starting
- Environment workers initializing
- vLLM loading the model
- Hanabi games being played
- Metrics logged to wandb

### Troubleshooting

#### Issue 1: vLLM Version Mismatch
**Error:** `NotImplementedError: roll vllm version 0.8.5.post1 is not supported`

**Solution:**
```bash
# Uninstall current vLLM
uv pip uninstall vllm

# Install exactly version 0.8.4
uv pip install vllm==0.8.4

# Verify
python -c "import vllm; print(vllm.__version__)"
```

#### Issue 2: Ray Worker Registration Failed
**Error:** `Failed to register worker to Raylet: IOError: Failed to read data`

**Solution:**
```bash
# Stop all Ray processes
ray stop --force
pkill -9 ray

# Clear Ray temp files
rm -rf /tmp/ray
rm -rf ~/.ray

# Unset stale environment variables
unset RAY_ADDRESS
unset RAY_REDIS_ADDRESS

# Try again
bash run_agentic_rollout_hanabi.sh
```

#### Issue 3: CUDA Out of Memory
**Error:** `torch.cuda.OutOfMemoryError`

**Solution:**
```bash
# In your config YAML, reduce gpu_memory_utilization
# Edit examples/hanabi/agentic_rollout_hanabi.yaml:
# gpu_memory_utilization: 0.45  # Try 0.3 or 0.2

# Or reduce batch size
# rollout_batch_size: 10  # Try 4 or 2
```

#### Issue 4: mcore_adapter Import Error
**Error:** `ModuleNotFoundError: No module named 'mcore_adapter'`

**Solution:**
```bash
cd ~/MARS-LLM/mcore_adapter
uv pip install -e .
cd ..
```

#### Issue 5: Missing OpenSpiel
**Error:** `ModuleNotFoundError: No module named 'pyspiel'`

**Solution:**
```bash
uv pip install open_spiel
```

### Performance Optimization Tips

1. **Use uv for faster package installation:**
   ```bash
   # uv is 10-100x faster than pip
   uv pip install package_name
   ```

2. **Enable Flash Attention 2:**
   Already configured in YAML files:
   ```yaml
   model_args:
     flash_attn: fa2  # Much faster than standard attention
   ```

3. **Adjust vLLM settings for your GPU:**
   ```yaml
   strategy_config:
     gpu_memory_utilization: 0.45  # Adjust based on GPU VRAM
     block_size: 16  # Larger = more memory, faster
     enable_prefix_caching: true  # Cache common prefixes
   ```

4. **Use multiple GPUs if available:**
   ```yaml
   num_gpus_per_node: 4  # Set to your GPU count
   device_mapping: list(range(0,4))  # Use GPUs 0-3
   ```

### Directory Structure After Setup

```
~/MARS-LLM/
├── .venv/                    # Virtual environment (created)
├── setup_env.sh              # Environment setup script (created)
├── runs/                     # Output directory (created during runs)
│   ├── hanabi_rollout/      # Rollout outputs
│   └── hanabi_selfplay/     # Training outputs
├── examples/                 # Example configs
├── roll/                     # Core framework
├── mcore_adapter/            # Megatron adapter
└── requirements_*.txt        # Dependency files
```

### Quick Reference Commands

```bash
# Activate environment
source .venv/bin/activate
source setup_env.sh

# Install package with uv
uv pip install package_name

# Update all packages
uv pip install --upgrade -r requirements_common.txt

# Clean Ray
ray stop --force && rm -rf /tmp/ray

# Run Hanabi evaluation
bash examples/hanabi/run_agentic_rollout_hanabi.sh

# Monitor with wandb
wandb online  # Ensure online mode

# Check GPU usage
watch -n 1 nvidia-smi
```

### Next Steps

After successful setup:
1. Read through the [Hanabi Example Walkthrough](#hanabi-example-walkthrough)
2. Review the [Configuration System](#configuration-system)
3. Run your first experiment: `bash examples/hanabi/run_agentic_rollout_hanabi.sh`
4. Check results in wandb: https://wandb.ai

---

## Project Structure

```
MARS-LLM/
├── roll/                          # Core framework
│   ├── agentic/                   # Agentic RL components
│   │   ├── env/                   # Game environments
│   │   │   ├── hanabi/           # Hanabi implementation
│   │   │   ├── tictactoe/
│   │   │   └── base.py           # Base environment classes
│   │   ├── rollout/              # Rollout scheduling
│   │   └── utils.py
│   ├── pipeline/                  # Pipeline orchestration
│   │   ├── agentic/
│   │   │   ├── agentic_pipeline.py      # Training pipeline
│   │   │   ├── agentic_rollout_pipeline.py  # Rollout-only pipeline
│   │   │   ├── agentic_config.py        # Configuration dataclasses
│   │   │   └── environment_worker.py    # Environment workers
│   │   └── base_pipeline.py
│   ├── models/                    # Model handling
│   ├── distributed/               # Distributed computing
│   ├── configs/                   # Base configurations
│   └── utils/                     # Utility functions
├── examples/                      # Example configurations and scripts
│   ├── hanabi/                   # Hanabi-specific examples
│   │   ├── agentic_val_hanabi_selfplay.yaml
│   │   ├── agentic_rollout_hanabi.yaml
│   │   ├── run_agentic_pipeline_hanabi_selfplay.sh
│   │   └── run_agentic_rollout_hanabi.sh
│   ├── config/                   # Shared configurations
│   │   ├── envs.yaml
│   │   ├── deepspeed_zero.yaml
│   │   └── ...
│   ├── start_agentic_pipeline.py      # Entry point for training
│   └── start_agentic_rollout_pipeline.py  # Entry point for rollout
├── data/                         # Training data
└── tests/                        # Test suite
```

---

## Core Components

### 1. **Environments** (`roll/agentic/env/`)
Abstract and concrete implementations of game environments.

### 2. **Pipelines** (`roll/pipeline/agentic/`)
Orchestrate training and rollout processes.

### 3. **Workers** (`roll/pipeline/agentic/environment_worker.py`)
Execute environment interactions in parallel.

### 4. **Rollout System** (`roll/agentic/rollout/`)
Manages scheduling and execution of game rollouts.

### 5. **Model Components** (`roll/models/`)
Handle model loading, inference, and training.

---

## Hanabi Example Walkthrough

Hanabi is a cooperative card game where players work together to build sequences of cards. This section demonstrates how MARS-LLM implements and trains on Hanabi.

### Directory Structure
```
examples/hanabi/
├── agentic_val_hanabi_selfplay.yaml    # Configuration for training with validation
├── agentic_rollout_hanabi.yaml         # Configuration for rollout-only mode
├── run_agentic_pipeline_hanabi_selfplay.sh  # Training script
└── run_agentic_rollout_hanabi.sh       # Rollout script
```

---

## Key Files and Their Roles

### 1. Entry Points

#### [`examples/start_agentic_pipeline.py`](examples/start_agentic_pipeline.py)
**Purpose:** Main entry point for training pipelines

**Key Functions:**
- Parses command-line arguments for config path and name
- Initializes Hydra configuration system
- Creates `AgenticConfig` from YAML configuration
- Instantiates and runs `AgenticPipeline`

**Code Flow:**
```python
parser.add_argument("--config_path", default="config")
parser.add_argument("--config_name", default="sppo_config")
initialize(config_path=args.config_path)
cfg = compose(config_name=args.config_name)
ppo_config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg))
pipeline = AgenticPipeline(pipeline_config=ppo_config)
pipeline.run()
```

#### [`examples/start_agentic_rollout_pipeline.py`](examples/start_agentic_rollout_pipeline.py)
**Purpose:** Entry point for rollout-only pipelines (no training)

**Difference from training pipeline:**
- Uses `AgenticRolloutPipeline` instead of `AgenticPipeline`
- Only performs environment rollouts without model updates
- Useful for evaluation and data collection

---

### 2. Environment Implementation

#### [`roll/agentic/env/base.py`](roll/agentic/env/base.py)
**Purpose:** Abstract base classes for all environments

**Key Classes:**

**`BaseEnv` (Abstract):**
```python
class BaseEnv(ABC):
    @abstractmethod
    def reset(self, seed=None) -> Any:
        """Reset environment with deterministic seed"""
        
    @abstractmethod
    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        """Execute action, return (observation, reward, done, info)"""
```

**`BaseDiscreteActionEnv` (Abstract):**
- Extends `BaseEnv` for discrete action spaces
- Adds `get_all_actions()` method
- Used by Hanabi, TicTacToe, etc.

**`BaseLanguageBasedEnv` (Abstract):**
- For environments with text-based actions
- Action space is strings instead of integers

---

#### [`roll/agentic/env/hanabi/config.py`](roll/agentic/env/hanabi/config.py)
**Purpose:** Configuration dataclass for Hanabi environment

**Key Parameters:**
```python
@dataclass
class HanabiConfig:
    seed: int = 42                    # Random seed for reproducibility
    render_mode: str = "text"         # Rendering mode (text/visual)
    built_in_opponent: str = "none"   # Opponent type (none/random)
    opponent_player: int = 1          # Which player is opponent
    
    # Game mechanics
    players: int = 2                  # Number of players
    colors: int = 2                   # Number of card colors
    ranks: int = 2                    # Number of card ranks
    hand_size: int = 3                # Cards per player
    max_information_tokens: int = 3   # Hint tokens available
    max_life_tokens: int = 3          # Mistakes allowed
    history_size: int = 4             # Game state history length
```

**Usage in YAML:**
```yaml
custom_envs:
  SimpleHanabi:
    env_type: hanabi
    env_config:
      render_mode: text
      colors: 3
      ranks: 2
      hand_size: 5
```

---

#### [`roll/agentic/env/hanabi/env.py`](roll/agentic/env/hanabi/env.py)
**Purpose:** Complete implementation of Hanabi game environment using OpenSpiel

**Key Components:**

**Initialization:**
```python
class Hanabi(BaseDiscreteActionEnv):
    def __init__(self, config: HanabiConfig = HanabiConfig()):
        self.config = config
        self.game_parameters = {
            "players": self.players,
            "colors": self.colors,
            "ranks": self.ranks,
            "hand_size": self.hand_size,
            "max_information_tokens": self.max_information_tokens,
            "max_life_tokens": self.max_life_tokens,
        }
        self._env = pyspiel.load_game("hanabi", self.game_parameters)
```

**Reset Method:**
- Seeds the environment for reproducibility
- Handles chance nodes (random card dealing)
- Returns initial observation and legal actions
- Supports built-in opponent for single-agent mode

**Step Method:**
```python
def step(self, action):
    execute_results = []
    current_player = self.current_player
    observation, rewards, done, info = self._step(action)
    
    execute_results.append({
        'current_player': current_player,
        'action': self._action_to_string(current_player, action),
        'rewards': rewards,
        'done': done,
        'info': info,
        'next_player': self.current_player,
        'observation': observation,
        'legal_actions': self.get_all_actions(),
    })
    return execute_results
```

**Prompt Generation:**
- `get_prompt()`: Returns system prompts for the LLM
- Includes game rules, current state, and valid actions
- Customizable for thinking mode vs direct action mode

**Action Handling:**
- Converts between string actions and integer action IDs
- Validates actions against legal moves
- Handles invalid actions with penalties

---

### 3. Pipeline Components

#### [`roll/pipeline/agentic/agentic_config.py`](roll/pipeline/agentic/agentic_config.py)
**Purpose:** Complete configuration system for agentic RL

**Key Configuration Classes:**

**`EnvManagerConfig`:**
```python
@dataclass
class EnvManagerConfig(WorkerConfig):
    env_groups: int = 128              # Number of environment groups
    group_size: int = 1                # Environments per group
    tags: List[str]                    # Environment type tags
    n_groups: List[int]                # Distribution of env types
    max_traj_per_env: int = -1        # Trajectories per environment
    format_penalty: float = 0         # Penalty for format errors
    max_env_num_per_worker: int = 0   # Parallelization control
```

**`RewardNormalizationConfig`:**
```python
@dataclass
class RewardNormalizationConfig:
    grouping: str = "state"           # Normalization scope (state/batch/inductive)
    method: str = "identity"          # Normalization method
    separate_norm_for_selfplay: bool  # Separate norms for each player
```

**`AgenticConfig` (Main Configuration):**
```python
@dataclass
class AgenticConfig(BaseConfig):
    # Environment configuration
    custom_envs: Dict[str, Any]           # Environment definitions
    train_env_manager: EnvManagerConfig   # Training environments
    val_env_manager: EnvManagerConfig     # Validation environments
    
    # Model paths
    pretrain: str                         # Base model path
    actor_train: WorkerConfig            # Training configuration
    actor_infer: WorkerConfig            # Inference configuration
    reference: WorkerConfig              # Reference model config
    
    # RL hyperparameters
    ppo_epochs: int = 1                  # PPO update epochs
    gamma: float = 1                     # Discount factor
    lambd: float = 0.95                  # GAE lambda
    pg_clip: float = 0.2                 # PPO clip range
    init_kl_coef: float = 0.2           # Initial KL coefficient
    whiten_advantages: bool = True       # Advantage whitening
    
    # Training settings
    rollout_batch_size: int              # Samples per rollout
    max_steps: int                       # Total training steps
    save_steps: int                      # Checkpoint frequency
    eval_steps: int                      # Evaluation frequency
    
    # Agentic-specific
    enable_response_mask: bool = True    # Mask system prompts
    action_sep: str = "||"              # Action separator
    enable_think: bool = True            # Enable thinking tokens
    use_turn_scores: bool                # Turn-level vs token-level rewards
```

---

#### [`roll/pipeline/agentic/agentic_pipeline.py`](roll/pipeline/agentic/agentic_pipeline.py)
**Purpose:** Main training pipeline orchestrating all components

**Architecture:**
```
AgenticPipeline
├── Actor Train Cluster     (model training)
├── Actor Infer Cluster     (rollout generation)
├── Reference Cluster       (KL penalty computation)
├── Critic Cluster          (value estimation, optional)
├── Train Rollout Scheduler (training rollouts)
└── Val Rollout Scheduler   (evaluation rollouts)
```

**Key Methods:**

**`__init__`:**
- Initializes all clusters (actor, reference, critic)
- Sets up rollout schedulers for training and validation
- Configures model update pairs
- Establishes checkpoint management

**`run()` - Main Training Loop:**
```python
@torch.no_grad()
def run(self):
    for global_step in range(self.pipeline_config.max_steps):
        # 1. Model update from training to inference
        model_update_metrics = self.model_update(global_step)
        
        # 2. Validation rollouts (periodic)
        if global_step % self.pipeline_config.eval_steps == 0:
            eval_batch = self.val_rollout_scheduler.get_batch(...)
            
        # 3. Training rollouts
        batch = self.train_rollout_scheduler.get_batch(...)
        
        # 4. Compute reference log probabilities (KL penalty)
        ref_log_probs = self.reference.compute_log_probs(batch)
        
        # 5. Compute values (if using GAE)
        if self.pipeline_config.adv_estimator == "gae":
            values = self.critic.compute_values(batch)
        
        # 6. Reward postprocessing
        batch = reward_postprocess_agentic(batch, self.pipeline_config)
        
        # 7. Apply KL penalty to rewards
        batch = apply_kl_penalty(batch, self.kl_ctrl)
        
        # 8. Compute advantages
        batch = compute_advantage(batch, self.pipeline_config)
        
        # 9. PPO training epochs
        for _ in range(self.pipeline_config.ppo_epochs):
            train_metrics = self.actor_train.train(batch)
            
        # 10. Log metrics
        self.tracker.log(metrics, step=global_step)
```

---

#### [`roll/pipeline/agentic/agentic_rollout_pipeline.py`](roll/pipeline/agentic/agentic_rollout_pipeline.py)
**Purpose:** Simplified pipeline for rollout-only execution (no training)

**Use Cases:**
- Model evaluation
- Data collection
- Debugging environment interactions
- Testing inference performance

**Simplified Flow:**
```python
@torch.no_grad()
def run(self):
    for global_step in range(self.pipeline_config.max_steps):
        # 1. Rollout generation only
        batch = self.rollout_scheduler.get_batch(...)
        
        # 2. Save rendered frames (if configured)
        if self.pipeline_config.render_save_dir:
            dump_rollout_render(...)
            
        # 3. Log metrics
        self.tracker.log(metrics, step=global_step)
```

---

#### [`roll/pipeline/agentic/environment_worker.py`](roll/pipeline/agentic/environment_worker.py)
**Purpose:** Distributed worker managing multiple environment instances

**Architecture:**
```
EnvironmentWorker
├── Multiple EnvManager instances (one per environment)
├── ThreadPoolExecutor (parallel execution)
├── Input Queue (prompts from model)
└── Output Queue (environment responses)
```

**Key Features:**

**Thread-Based Parallelism:**
```python
def run_rollout_loop(self, data: DataProto):
    with ThreadPoolExecutor(max_workers=len(self.env_managers)) as executor:
        futures_list = [
            executor.submit(env_manager.run_rollout_loop, data)
            for env_id, env_manager in self.env_managers.items()
        ]
```

**Environment Group Management:**
- Each worker manages multiple environments
- Environments in same group share config and seed
- Enables efficient batching and parallelization

**Initialization:**
```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def initialize(self, pipeline_config, generate_scheduler, 
               input_queue, output_queue, collator, mode):
    # Create one EnvManager per environment
    for env_id, env_config in self.env_configs.items():
        self.env_managers[env_id] = EnvManager(...)
```

---

### 4. Configuration Files

#### [`examples/hanabi/agentic_val_hanabi_selfplay.yaml`](examples/hanabi/agentic_val_hanabi_selfplay.yaml)
**Purpose:** Complete configuration for Hanabi training with self-play

**Configuration Breakdown:**

**1. Basic Settings:**
```yaml
exp_name: "agentic_pipeline"
seed: 42
num_gpus_per_node: 8
max_steps: 200              # Total training iterations
save_steps: 20              # Checkpoint every 20 steps
eval_steps: 5               # Evaluate every 5 steps
```

**2. Batch Sizes:**
```yaml
rollout_batch_size: 128     # Training rollouts per step
val_batch_size: 1500        # Validation rollouts per step
sequence_length: 32768      # Maximum sequence length
```

**3. RL Hyperparameters:**
```yaml
ppo_epochs: 1               # PPO update epochs
adv_estimator: "reinforce"  # Advantage estimation method
init_kl_coef: 0.0          # Initial KL coefficient
use_kl_loss: true
kl_loss_coef: 0.20
whiten_advantages: true
dual_clip_loss: true
pg_clip_high: 0.20
```

**4. Model Configurations:**

**Actor Training:**
```yaml
actor_train:
  model_args:
    flash_attn: fa2
    dtype: bf16
  training_args:
    learning_rate: 1e-6
    weight_decay: 0.05
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 2
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 4
      sequence_parallel: true
```

**Actor Inference:**
```yaml
actor_infer:
  generating_args:
    max_new_tokens: 4096
    top_p: 0.99
    top_k: 100
    temperature: 0.6
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.8
```

**5. Environment Configuration:**
```yaml
enable_response_mask: True
action_sep: "||"
use_turn_scores: True
enable_think: True
max_actions_per_traj: 50

custom_envs:
  TicTacToe-first-100:
    env_type: tictactoe
    max_actions_per_traj: 50
    grid_vocab: false
    env_config:
      built_in_opponent: none
      opponent_player: 1
      
train_env_manager:
  format_penalty: 0.05
  max_env_num_per_worker: 64
  env_groups: 48
  group_size: 1
  tags: [TicTacToe-first-100]
```

**6. Reward Normalization:**
```yaml
reward_normalization:
  grouping: tags
  method: mean
  separate_norm_for_selfplay: true
whiten_rewards: true
advantage_norm: mean
```

---

#### [`examples/hanabi/agentic_rollout_hanabi.yaml`](examples/hanabi/agentic_rollout_hanabi.yaml)
**Purpose:** Configuration for rollout-only mode (evaluation/data collection)

**Key Differences from Training Config:**

**1. Simplified Settings:**
```yaml
max_steps: 1                # Usually just one rollout step
save_steps: 10000           # No frequent checkpoints needed
eval_steps: 10              # No training updates
```

**2. Multiple Environment Variants:**
```yaml
custom_envs:
  MiniHanabi:
    env_type: hanabi
    env_config:
      render_mode: text
      colors: 2
      ranks: 2
      hand_size: 3
      
  SimpleHanabi:
    env_type: hanabi
    env_config:
      colors: 3
      ranks: 2
      hand_size: 5
      max_information_tokens: 8
      
  FullHanabi:
    env_type: hanabi
    env_config:
      colors: 3
      ranks: 3
      hand_size: 5
```

**3. Environment Distribution:**
```yaml
train_env_manager:
  env_groups: 48
  group_size: 1
  tags: [MiniHanabi, SimpleHanabi, FullHanabi]
  n_groups: [16, 16, 16]    # Equal distribution
```

---

### 5. Execution Scripts

#### [`examples/hanabi/run_agentic_pipeline_hanabi_selfplay.sh`](examples/hanabi/run_agentic_pipeline_hanabi_selfplay.sh)
**Purpose:** Shell script to launch training pipeline

**Script Breakdown:**
```bash
#!/bin/bash
set +x
ray stop                    # Clean up any existing Ray processes

CONFIG_PATH=$(basename $(dirname $0))  # Get config directory name

# Set Python path
ROLL_PATH=${PWD}
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

# Create output directories
ROLL_OUTPUT_DIR="./runs/hanabi_selfplay/$(date +%Y%m%d-%H%M%S)"
ROLL_LOG_DIR=$ROLL_OUTPUT_DIR/logs
ROLL_RENDER_DIR=$ROLL_OUTPUT_DIR/render
export ROLL_OUTPUT_DIR=$ROLL_OUTPUT_DIR
export ROLL_LOG_DIR=$ROLL_LOG_DIR
export ROLL_RENDER_DIR=$ROLL_RENDER_DIR
mkdir -p $ROLL_LOG_DIR $ROLL_RENDER_DIR

# Launch training
python examples/start_agentic_pipeline.py \
    --config_path $CONFIG_PATH \
    --config_name agentic_val_hanabi_selfplay \
    | tee $ROLL_LOG_DIR/custom_logs.log
```

**Key Features:**
- Automatic timestamped output directories
- Logging to both console and file
- Ray cluster management
- Environment variable setup

---

#### [`examples/hanabi/run_agentic_rollout_hanabi.sh`](examples/hanabi/run_agentic_rollout_hanabi.sh)
**Purpose:** Shell script to launch rollout-only pipeline

**Differences:**
```bash
# Output to different directory
ROLL_OUTPUT_DIR="./runs/hanabi_rollout/$(date +%Y%m%d-%H%M%S)"

# Use rollout pipeline entry point
python examples/start_agentic_rollout_pipeline.py \
    --config_path $CONFIG_PATH \
    --config_name agentic_rollout_hanabi \
    | tee $ROLL_LOG_DIR/custom_logs.log
```

---

## Pipeline Flow

### Training Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Start Training Step                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Model Update: Actor Train → Actor Infer                     │
│     (Sync trained weights to inference model)                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Validation Rollout (if eval_step)                           │
│     ├─ Generate game trajectories with current policy           │
│     ├─ Compute episode scores                                   │
│     └─ Log evaluation metrics                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Training Rollout                                             │
│     ├─ EnvironmentWorker: Manage multiple game instances        │
│     ├─ EnvManager: Execute game loops                           │
│     │   ├─ Get LLM response from actor_infer                    │
│     │   ├─ Parse action from response                           │
│     │   ├─ Execute action in environment                        │
│     │   └─ Collect rewards and next state                       │
│     └─ Collect trajectories into batch                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Reference Model Log Probabilities                            │
│     ├─ Compute log probs for reference policy                   │
│     └─ Used for KL divergence penalty                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Value Estimation (if using GAE)                             │
│     ├─ Critic computes state values                             │
│     └─ Used for advantage calculation                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. Reward Postprocessing                                        │
│     ├─ Reward normalization (per group/batch/state)             │
│     ├─ Reward whitening                                         │
│     ├─ Format penalty application                               │
│     └─ Turn-level vs token-level reward assignment              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. Apply KL Penalty                                             │
│     ├─ Compute KL divergence from reference                     │
│     └─ Adjust rewards: r' = r - β * KL                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  8. Compute Advantages                                           │
│     ├─ If REINFORCE: A = R (returns)                           │
│     ├─ If GAE: A = Σ(γλ)^t (r + γV(s') - V(s))               │
│     └─ Whiten advantages if configured                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  9. PPO Training Loop                                            │
│     For each epoch:                                              │
│       ├─ Compute current policy log probs                       │
│       ├─ Compute policy ratio: r = π_new / π_old               │
│       ├─ Clip policy ratio                                      │
│       ├─ Compute policy loss: L_CLIP                            │
│       ├─ Compute value loss (if using critic)                   │
│       ├─ Compute entropy bonus                                  │
│       ├─ Backward pass and optimization                         │
│       └─ Update KL controller                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  10. Checkpoint & Logging                                        │
│      ├─ Save model checkpoint (if save_step)                    │
│      ├─ Log metrics to TensorBoard/WandB                        │
│      └─ Update global step counter                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
                   Next Training Step
```

---

### Rollout Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Start Rollout Step                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Environment Initialization                                   │
│     ├─ Load environment configurations                           │
│     ├─ Create environment instances                              │
│     └─ Set random seeds for reproducibility                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Parallel Rollout Execution                                   │
│     For each environment instance:                               │
│       ├─ Reset environment                                       │
│       ├─ Get initial observation                                 │
│       ├─ Generate prompt from observation                        │
│       ├─ Query actor_infer for action                           │
│       ├─ Parse and validate action                              │
│       ├─ Execute action in environment                           │
│       ├─ Collect reward and next observation                     │
│       └─ Repeat until episode done                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Data Collection                                              │
│     ├─ Aggregate trajectories from all environments              │
│     ├─ Collect episode scores                                    │
│     ├─ Save rendered frames (if configured)                      │
│     └─ Compute rollout metrics                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Logging & Visualization                                      │
│     ├─ Log rollout metrics                                       │
│     ├─ Save trajectory visualizations                            │
│     ├─ Print sample episodes                                     │
│     └─ Update statistics                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
                   Next Rollout Step
```

---

## Configuration System

### Hydra Configuration Hierarchy

MARS-LLM uses Hydra for hierarchical configuration management:

```
examples/hanabi/agentic_val_hanabi_selfplay.yaml (Main Config)
├── defaults:
│   ├── ../config/envs.yaml                (Environment definitions)
│   ├── ../config/deepspeed_zero.yaml      (DeepSpeed ZeRO settings)
│   ├── ../config/deepspeed_zero2.yaml
│   ├── ../config/deepspeed_zero3.yaml
│   └── ../config/deepspeed_zero3_cpuoffload.yaml
├── exp_name, seed, logging settings
├── model configurations (actor, critic, reference)
├── custom_envs (environment definitions)
├── train_env_manager (training environment settings)
└── val_env_manager (validation environment settings)
```

### Configuration Override

You can override any parameter:
```bash
python examples/start_agentic_pipeline.py \
    --config_path hanabi \
    --config_name agentic_val_hanabi_selfplay \
    max_steps=500 \
    rollout_batch_size=256 \
    actor_train.training_args.learning_rate=5e-7
```

---

## Environment System

### Environment Registration

Environments are registered in `roll/agentic/env/__init__.py`:
```python
REGISTERED_ENVS = {
    "hanabi": Hanabi,
    "tictactoe": TicTacToe,
    "connect_four": ConnectFour,
    # ... other environments
}
```

### Environment Lifecycle

```
1. Configuration → HanabiConfig created from YAML
2. Instantiation → Hanabi(config) in EnvManager
3. Reset → env.reset(seed) initializes game state
4. Loop:
   ├─ env.get_prompt() → generates LLM prompt
   ├─ model.generate() → LLM produces action
   ├─ env.step(action) → executes action
   └─ Check if done
5. Episode ends → collect rewards and metrics
```

### Multi-Agent Coordination

For multi-agent games like Hanabi:
```python
# Environment tracks current player
current_player = env.current_player

# Self-play mode: both players use same model
if current_player == 0:
    action = model.generate(prompt_player0)
else:
    action = model.generate(prompt_player1)
    
# Built-in opponent mode: one player uses fixed policy
if current_player == opponent_player:
    action = env._opponent_step()  # Random or rule-based
else:
    action = model.generate(prompt)
```

---

## Model Components

### Model Roles

**1. Actor (Training):**
- Updated via PPO
- Uses DeepSpeed/Megatron for distributed training
- Gradient checkpointing enabled

**2. Actor (Inference):**
- Periodically synced from Actor (Training)
- Uses vLLM for fast inference
- No gradients computed

**3. Reference:**
- Frozen copy of initial model
- Computes KL divergence penalty
- Prevents policy from diverging too far

**4. Critic (Optional):**
- Value function estimator
- Used with GAE advantage estimation
- Also trained via PPO

### Model Update Flow

```
Training Step N:
1. Actor (Train) weights updated via PPO
2. Actor (Infer) still using weights from step N-k

Training Step N+k:
1. Actor (Train) → Actor (Infer) sync
2. New rollouts use updated policy
```

---

## Running Examples

### 1. Training Hanabi with Self-Play

```bash
cd /home/dematsunaga/MARS-LLM

# Launch training
bash examples/hanabi/run_agentic_pipeline_hanabi_selfplay.sh

# Monitor training
tensorboard --logdir=./runs/hanabi_selfplay/
```

**Expected Output:**
- Checkpoints in `./runs/hanabi_selfplay/<timestamp>/`
- TensorBoard logs with metrics
- Periodic evaluation results

### 2. Rollout-Only Evaluation

```bash
# Run rollouts without training
bash examples/hanabi/run_agentic_rollout_hanabi.sh

# Check results in output directory
ls ./runs/hanabi_rollout/<timestamp>/
```

### 3. Custom Configuration

Create your own config `examples/hanabi/my_hanabi_config.yaml`:
```yaml
defaults:
  - agentic_val_hanabi_selfplay

# Override specific parameters
max_steps: 100
rollout_batch_size: 64

custom_envs:
  MyHanabi:
    env_type: hanabi
    env_config:
      colors: 4
      ranks: 3
      hand_size: 4
```

Run with:
```bash
python examples/start_agentic_pipeline.py \
    --config_path hanabi \
    --config_name my_hanabi_config
```

---

## Key Concepts

### 1. **Environment Groups**
- Environments in same group share config and seed
- Ensures reproducibility and controlled experiments
- Example: `env_groups=48, group_size=1` = 48 unique environments

### 2. **Turn Scores vs Token Scores**
- `use_turn_scores=True`: Reward entire turn based on action outcome
- `use_turn_scores=False`: Token-level credit assignment
- Critical for proper credit assignment in multi-turn games

### 3. **Think Tokens**
- `enable_think=True`: LLM can use `<think>...</think>` for reasoning
- Thinking process not penalized in RL objective
- Only action tokens receive gradient updates

### 4. **Reward Normalization**
- `grouping`: Normalize within groups (tags/batch/state)
- `separate_norm_for_selfplay`: Independent norms for each player
- Stabilizes training across diverse game outcomes

### 5. **Advantage Estimation**
- `reinforce`: Simple returns-based (Monte Carlo)
- `gae`: Generalized Advantage Estimation (requires critic)
- GAE reduces variance but adds complexity

---

## Debugging Tips

### 1. Check Environment Rollouts
```bash
# Run rollout pipeline with small batch
python examples/start_agentic_rollout_pipeline.py \
    --config_path hanabi \
    --config_name agentic_rollout_hanabi \
    rollout_batch_size=4
```

### 2. Enable Rendering
```yaml
render_save_dir: ./runs/renders
```
Saves game visualizations to inspect trajectories

### 3. Check Action Parsing
```python
# In environment logs, look for:
# - "Invalid action" messages
# - format_penalty applications
# - Action distribution statistics
```

### 4. Monitor Metrics
Key metrics to watch:
- `score/mean`: Average episode score
- `rollout/format_error_rate`: Action parsing failures
- `train/policy_loss`: PPO policy loss
- `train/clip_fraction`: Fraction of clipped updates
- `critic/ref_log_prob`: KL divergence indicator

---

## Advanced Topics

### 1. Multi-Game Training
Combine multiple games in one config:
```yaml
custom_envs:
  SimpleHanabi:
    env_type: hanabi
  TicTacToe:
    env_type: tictactoe
    
train_env_manager:
  tags: [SimpleHanabi, TicTacToe]
  n_groups: [32, 32]  # 50% each
```

### 2. Distributed Training
Scale across multiple nodes:
```yaml
num_gpus_per_node: 8
actor_train:
  strategy_args:
    tensor_model_parallel_size: 4
    pipeline_model_parallel_size: 2
```

### 3. Custom Environments
Implement `BaseDiscreteActionEnv`:
```python
class MyGame(BaseDiscreteActionEnv):
    def reset(self, seed=None):
        # Initialize game
        return observation, execute_results
        
    def step(self, action):
        # Execute action
        return execute_results
        
    def get_all_actions(self):
        # Return valid actions
        return action_dict
        
    def get_prompt(self, mode="prefix"):
        # Generate LLM prompt
        return prompt_string
```

---

## Summary

This guide covers the essential components of the MARS-LLM codebase using Hanabi as a concrete example. Key takeaways:

1. **Modular Architecture**: Environments, pipelines, and workers are cleanly separated
2. **Flexible Configuration**: Hydra enables easy experimentation
3. **Distributed Execution**: Ray + DeepSpeed scale training efficiently
4. **Multi-Agent RL**: Self-play and opponent modes supported
5. **Extensible Design**: Easy to add new games and training algorithms

For more examples, explore:
- [`examples/tictactoe/`](examples/tictactoe/) - Simpler game for learning
- [`examples/connect_four/`](examples/connect_four/) - More complex strategy
- [`examples/kuhn_poker/`](examples/kuhn_poker/) - Imperfect information game

Happy training! 🚀
