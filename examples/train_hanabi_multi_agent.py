#!/usr/bin/env python3
"""
Multi-Agent LoRA Training with Real Hanabi Environment

This script performs actual multi-agent RL training on Hanabi without Ray.
Each agent has a separate LoRA adapter that only updates on its own trajectories.

Usage:
    python examples/train_hanabi_multi_agent.py \
        --model Qwen/Qwen3-0.6B \
        --num_episodes 10 \
        --num_epochs 3 \
        --output_dir ./output/hanabi_training
"""

import argparse
import datetime
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/MARS")

from roll.agentic.env.hanabi import Hanabi, HanabiConfig
from roll.multi_agent import MultiAgentConfig, MultiAgentLoRAModel
from roll.utils.logging import get_logger

logger = get_logger()


class TrajectoryDataset(Dataset):
    """Dataset of trajectories collected from Hanabi episodes."""
    
    def __init__(
        self, 
        trajectories: List[Dict], 
        tokenizer, 
        max_length: int = 1024
    ):
        self.trajectories = trajectories
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        
        # Build prompt + response
        prompt = traj["prompt"]
        response = traj["response"]
        full_text = prompt + response
        
        # Tokenize
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Create labels (mask prompt, only train on response)
        prompt_encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_length = prompt_encoded["input_ids"].shape[1]
        
        labels = encoded["input_ids"].clone()
        labels[:, :prompt_length] = -100  # Mask prompt tokens
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "agent_id": traj["agent_id"],
            "reward": traj["reward"],
        }


def parse_action_from_response(response: str) -> Optional[str]:
    """Extract action from LLM response."""
    # Look for <answer>...</answer> pattern
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def generate_response(
    model: MultiAgentLoRAModel,
    tokenizer,
    prompt: str,
    agent_id: int,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    device: str = "cuda",
) -> str:
    """Generate a response from the model for a given prompt."""
    model.set_active_agent(agent_id)
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def collect_episode(
    env: Hanabi,
    model: MultiAgentLoRAModel,
    tokenizer,
    seed: int,
    max_turns: int = 50,
    device: str = "cuda",
) -> Tuple[List[Dict], Dict]:
    """
    Collect a single episode of Hanabi gameplay.
    
    Returns:
        trajectories: List of trajectory dicts with prompt, response, agent_id, reward
        info: Episode summary info
    """
    trajectories = []
    
    # Reset environment
    initial_obs, execute_results = env.reset(seed=seed)
    
    observation = execute_results[-1]['observation'] if execute_results else initial_obs['observation']
    legal_actions = execute_results[-1]['legal_actions'] if execute_results else initial_obs['legal_actions']
    done = execute_results[-1]['done'] if execute_results else False
    
    turn = 0
    while not done and turn < max_turns:
        current_player = env.current_player
        
        # Build prompt
        prefix_prompt = env.get_prompt(mode="prefix", think=True, player_id=current_player)
        legal_actions_str = "\n".join([f"  - {v}" for v in legal_actions.values()])
        
        prompt = (
            f"{prefix_prompt['system']}\n\n"
            f"{prefix_prompt['user']}\n\n"
            f"CURRENT GAME STATE:\n{observation}\n\n"
            f"LEGAL ACTIONS:\n{legal_actions_str}\n\n"
            f"Think step by step and choose your action:"
        )
        
        # Generate response from agent's adapter
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            agent_id=current_player,
            device=device,
        )
        
        # Parse action from response
        action_str = parse_action_from_response(response)
        
        if action_str is None:
            # Invalid format - penalize and end episode
            execute_results = env.get_losing_state(player_id=current_player)
            reward = execute_results[-1]['rewards'][current_player]
            done = True
        else:
            # Try to execute the action
            try:
                # Find matching action
                matched_action = None
                for action_val in legal_actions.values():
                    if action_str.lower() in action_val.lower() or action_val.lower() in action_str.lower():
                        matched_action = action_val
                        break
                
                if matched_action is None:
                    # No matching action - take random legal action
                    import random
                    matched_action = random.choice(list(legal_actions.values()))
                
                execute_results = env.step(matched_action)
                reward = execute_results[-1]['rewards'][current_player]
                done = execute_results[-1]['done']
                
                if not done:
                    observation = execute_results[-1]['observation']
                    legal_actions = execute_results[-1]['legal_actions']
                    
            except Exception as e:
                logger.warning(f"Error executing action: {e}")
                execute_results = env.get_losing_state(player_id=current_player)
                reward = execute_results[-1]['rewards'][current_player]
                done = True
        
        # Store trajectory
        trajectories.append({
            "prompt": prompt,
            "response": response,
            "agent_id": current_player,
            "reward": reward,
            "turn": turn,
        })
        
        turn += 1
    
    # Get final returns
    info = execute_results[-1].get('info', {}) if execute_results else {}
    episode_return = info.get(f'player_0_return', 0) + info.get(f'player_1_return', 0)
    
    return trajectories, {
        "total_turns": turn,
        "episode_return": episode_return,
        "player_0_return": info.get('player_0_return', 0),
        "player_1_return": info.get('player_1_return', 0),
    }


def compute_policy_gradient_loss(
    model: MultiAgentLoRAModel,
    batch: Dict,
    agent_id: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute policy gradient loss for a batch of trajectories.
    
    Uses reward-weighted cross-entropy loss (REINFORCE).
    """
    model.set_active_agent(agent_id)
    model.train()
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    rewards = batch["reward"].to(device).float()
    
    # Normalize rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute per-token loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    per_token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(shift_labels.size())
    
    # Mask and sum per-sequence loss
    mask = (shift_labels != -100).float()
    per_seq_loss = (per_token_loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    
    # Reward-weighted loss
    weighted_loss = (per_seq_loss * rewards).mean()
    
    return weighted_loss


def train_epoch(
    model: MultiAgentLoRAModel,
    optimizer: torch.optim.Optimizer,
    trajectories: List[Dict],
    tokenizer,
    num_agents: int,
    batch_size: int = 4,
    max_length: int = 1024,
    device: str = "cuda",
) -> Dict[str, float]:
    """Train for one epoch on collected trajectories."""
    metrics = defaultdict(list)
    
    # Partition trajectories by agent
    agent_trajectories = {i: [] for i in range(num_agents)}
    for traj in trajectories:
        agent_id = traj["agent_id"]
        if agent_id < num_agents:
            agent_trajectories[agent_id].append(traj)
    
    # Train each agent's adapter on its trajectories
    for agent_id in range(num_agents):
        agent_trajs = agent_trajectories[agent_id]
        if len(agent_trajs) == 0:
            continue
        
        # Create dataset and dataloader
        dataset = TrajectoryDataset(agent_trajs, tokenizer, max_length)
        dataloader = DataLoader(
            dataset, 
            batch_size=min(batch_size, len(agent_trajs)),
            shuffle=True,
            collate_fn=lambda x: {
                "input_ids": torch.stack([item["input_ids"] for item in x]),
                "attention_mask": torch.stack([item["attention_mask"] for item in x]),
                "labels": torch.stack([item["labels"] for item in x]),
                "agent_id": torch.tensor([item["agent_id"] for item in x]),
                "reward": torch.tensor([item["reward"] for item in x]),
            }
        )
        
        agent_losses = []
        for batch in dataloader:
            optimizer.zero_grad()
            loss = compute_policy_gradient_loss(model, batch, agent_id, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            agent_losses.append(loss.item())
        
        metrics[f"agent_{agent_id}/loss"] = sum(agent_losses) / len(agent_losses)
        metrics[f"agent_{agent_id}/num_samples"] = len(agent_trajs)
    
    return dict(metrics)


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent LoRA Training on Hanabi")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Base model")
    parser.add_argument("--num_episodes", type=int, default=10, help="Episodes per training iteration")
    parser.add_argument("--num_iterations", type=int, default=5, help="Training iterations")
    parser.add_argument("--num_epochs", type=int, default=3, help="Epochs per iteration")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--output_dir", type=str, default="./output/hanabi_training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("Multi-Agent LoRA Training on Hanabi")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Agents: {args.num_agents}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Episodes per iteration: {args.num_episodes}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    
    # Create multi-agent LoRA model
    print("Creating multi-agent LoRA model...")
    multi_agent_config = MultiAgentConfig(
        enabled=True,
        num_agents=args.num_agents,
        default_lora_rank=args.lora_rank,
        default_lora_alpha=args.lora_rank,
        default_target_modules="q_proj,k_proj,v_proj,o_proj",
    )
    model = MultiAgentLoRAModel(base_model, multi_agent_config).to(device)
    
    # Freeze base model, only train LoRA
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    
    # Create Hanabi environment (MiniHanabi config)
    print("\nCreating Hanabi environment...")
    env_config = HanabiConfig(
        players=2,
        colors=2,  # Mini Hanabi
        ranks=3,
        hand_size=3,
        max_information_tokens=4,
        max_life_tokens=2,
    )
    env = Hanabi(config=env_config)
    
    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    all_metrics = []
    
    for iteration in range(args.num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{args.num_iterations}")
        print("=" * 60)
        
        # Collect episodes
        print(f"\nCollecting {args.num_episodes} episodes...")
        all_trajectories = []
        episode_returns = []
        
        for ep in tqdm(range(args.num_episodes), desc="Collecting episodes"):
            seed = args.seed + iteration * 1000 + ep
            trajs, info = collect_episode(
                env=env,
                model=model,
                tokenizer=tokenizer,
                seed=seed,
                device=str(device),
            )
            all_trajectories.extend(trajs)
            episode_returns.append(info["episode_return"])
        
        print(f"Collected {len(all_trajectories)} trajectory steps")
        print(f"Mean episode return: {sum(episode_returns)/len(episode_returns):.2f}")
        
        # Train on collected trajectories
        print(f"\nTraining for {args.num_epochs} epochs...")
        for epoch in range(args.num_epochs):
            epoch_metrics = train_epoch(
                model=model,
                optimizer=optimizer,
                trajectories=all_trajectories,
                tokenizer=tokenizer,
                num_agents=args.num_agents,
                batch_size=args.batch_size,
                max_length=args.max_length,
                device=str(device),
            )
            
            print(f"  Epoch {epoch + 1}: " + ", ".join(
                f"{k}={v:.4f}" for k, v in epoch_metrics.items()
            ))
        
        # Record metrics
        iteration_metrics = {
            "iteration": iteration,
            "mean_return": sum(episode_returns) / len(episode_returns),
            "num_trajectories": len(all_trajectories),
            **epoch_metrics,
        }
        all_metrics.append(iteration_metrics)
        
        # Save checkpoint every iteration
        ckpt_dir = os.path.join(args.output_dir, f"iteration_{iteration}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_all_adapters(ckpt_dir)
        print(f"Saved checkpoint to {ckpt_dir}")
    
    # Final save
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_all_adapters(final_dir)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nFinal model saved to: {final_dir}")
    print(f"Metrics saved to: {metrics_path}")
    
    # Summary
    print("\nTraining Summary:")
    for m in all_metrics:
        print(f"  Iteration {m['iteration']}: return={m['mean_return']:.2f}")


if __name__ == "__main__":
    main()

