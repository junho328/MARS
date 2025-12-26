"""
Multi-Agent Agentic Pipeline Launcher

Starts the multi-agent RL training pipeline where each agent
has its own dedicated LoRA adapter.

Usage:
    python examples/start_multi_agent_pipeline.py \
        --config_path hanabi \
        --config_name agentic_val_hanabi_selfplay \
        --num_agents 2
"""

import argparse

from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.multi_agent import MultiAgentConfig, create_multi_agent_pipeline
from roll.pipeline.agentic.agentic_config import AgenticConfig


def main():
    parser = argparse.ArgumentParser(
        description="Start multi-agent LoRA training pipeline"
    )
    parser.add_argument(
        "--config_path", 
        help="The path of the main configuration file", 
        default="config"
    )
    parser.add_argument(
        "--config_name", 
        help="The name of the main configuration file (without extension).", 
        default="sppo_config"
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=2,
        help="Number of agents (each with a separate LoRA adapter)"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="LoRA rank for each adapter"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha for each adapter"
    )
    parser.add_argument(
        "--lora_target",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated list of target modules for LoRA"
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="sequential",
        choices=["sequential", "parallel"],
        help="Training strategy: sequential (one adapter at a time) or parallel"
    )
    args = parser.parse_args()

    # Initialize Hydra and load config
    # Config path should be relative to this file's directory
    config_path = args.config_path
    if not config_path.startswith("../") and not config_path.startswith("/"):
        config_path = f"../{config_path}" if "/" in config_path else config_path
    
    initialize(config_path=config_path, job_name="multi_agent_app", version_base=None)
    cfg = compose(config_name=args.config_name)

    print("=" * 60)
    print("Multi-Agent LoRA Training Configuration")
    print("=" * 60)
    print(f"Number of agents: {args.num_agents}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print(f"Target modules: {args.lora_target}")
    print(f"Training strategy: {args.training_strategy}")
    print("=" * 60)
    print("\nPipeline Configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Create pipeline config
    ppo_config = from_dict(
        data_class=AgenticConfig, 
        data=OmegaConf.to_container(cfg, resolve=True)
    )

    # Create multi-agent config
    multi_agent_config = MultiAgentConfig(
        enabled=True,
        num_agents=args.num_agents,
        default_lora_rank=args.lora_rank,
        default_lora_alpha=args.lora_alpha,
        default_target_modules=args.lora_target,
        training_strategy=args.training_strategy,
    )

    # Initialize distributed infrastructure
    init()

    # Create and run the multi-agent pipeline
    pipeline = create_multi_agent_pipeline(
        pipeline_config=ppo_config,
        multi_agent_config=multi_agent_config,
    )

    print("\nStarting multi-agent training...")
    pipeline.run()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

