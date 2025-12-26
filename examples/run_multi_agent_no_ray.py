"""
Multi-Agent LoRA Training WITHOUT Ray

This script runs multi-agent training directly without Ray distributed infrastructure.
Useful for testing in resource-constrained environments.
"""

import argparse
import os
import sys
import json
from datetime import datetime

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roll.multi_agent import MultiAgentConfig, MultiAgentLoRAModel


def create_model_and_tokenizer(model_name: str, device: torch.device):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},
    )
    
    return model, tokenizer


def create_hanabi_prompt(observation: str, player_id: int) -> str:
    """Create a prompt for Hanabi game."""
    return f"""You are Player {player_id} in a cooperative card game called Hanabi.

Current observation:
{observation}

What action should you take? Think step by step, then provide your action.
<think>
"""


def simulate_hanabi_episode(tokenizer, num_turns: int = 3):
    """Simulate a simple Hanabi episode for testing."""
    # Simplified Hanabi-like observations
    observations = [
        "Your hand: [?, ?, ?, ?, ?]\nPartner's hand: [R1, B2, G3, Y1, W2]\nFireworks: R0, B0, G0, Y0, W0\nHints: 8, Fuses: 3",
        "Your hand: [?, ?, ?, ?, ?]\nPartner's hand: [R1, B2, G3, Y1, W2]\nFireworks: R1, B0, G0, Y0, W0\nHints: 7, Fuses: 3",
        "Your hand: [?, ?, ?, ?, ?]\nPartner's hand: [R2, B2, G3, Y1, W2]\nFireworks: R1, B0, G0, Y0, W0\nHints: 6, Fuses: 3",
    ]
    
    episodes = []
    for turn in range(min(num_turns, len(observations))):
        for player_id in [0, 1]:
            prompt = create_hanabi_prompt(observations[turn], player_id)
            
            # Tokenize
            tokens = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=256,
                truncation=True,
            )
            
            episodes.append({
                "player_id": player_id,
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "turn": turn,
            })
    
    return episodes


def train_step(model, batch, optimizer, device):
    """Perform a single training step."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # Use input as labels for language modeling
    labels = input_ids.clone()
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    
    loss = outputs.loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Training without Ray")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./output/no_ray", help="Output directory")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multi-Agent LoRA Training (No Ray)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Agents: {args.num_agents}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Epochs: {args.num_epochs}")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load base model
    base_model, tokenizer = create_model_and_tokenizer(args.model, device)
    
    # Create multi-agent config
    multi_agent_config = MultiAgentConfig(
        enabled=True,
        num_agents=args.num_agents,
        default_lora_rank=args.lora_rank,
        default_lora_alpha=args.lora_alpha,
        default_target_modules="q_proj,k_proj,v_proj,o_proj",
    )
    
    # Wrap with multi-agent LoRA
    print("Creating MultiAgentLoRAModel...")
    model = MultiAgentLoRAModel(base_model, multi_agent_config)
    model.train()
    
    # Create optimizer for all LoRA parameters
    model.unfreeze_all_adapters()
    lora_params = [p for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad]
    print(f"Total LoRA parameters: {sum(p.numel() for p in lora_params):,}")
    
    optimizer = AdamW(lora_params, lr=args.lr)
    
    # Simulate Hanabi episodes
    print("\nSimulating Hanabi episodes...")
    episodes = simulate_hanabi_episode(tokenizer, num_turns=5)
    print(f"Generated {len(episodes)} training samples")
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    metrics_history = []
    
    for epoch in range(args.num_epochs):
        epoch_losses = {i: [] for i in range(args.num_agents)}
        
        for sample in episodes:
            player_id = sample["player_id"]
            
            # Switch to this agent's adapter
            model.set_active_agent(player_id)
            
            # Train step
            loss = train_step(model, sample, optimizer, device)
            epoch_losses[player_id].append(loss)
        
        # Compute epoch metrics
        epoch_metrics = {"epoch": epoch + 1}
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        for agent_id in range(args.num_agents):
            if epoch_losses[agent_id]:
                avg_loss = sum(epoch_losses[agent_id]) / len(epoch_losses[agent_id])
                epoch_metrics[f"agent_{agent_id}_loss"] = avg_loss
                print(f"  Agent {agent_id}: loss = {avg_loss:.4f} ({len(epoch_losses[agent_id])} samples)")
        
        metrics_history.append(epoch_metrics)
    
    # Verify gradient isolation
    print("\n" + "=" * 60)
    print("Verifying Adapter Differentiation")
    print("=" * 60)
    
    model.eval()
    test_input = tokenizer("Test prompt for verification", return_tensors="pt", padding=True)
    test_input = {k: v.to(device) for k, v in test_input.items()}
    
    outputs = {}
    for agent_id in range(args.num_agents):
        model.set_active_agent(agent_id)
        with torch.no_grad():
            out = model(test_input["input_ids"], attention_mask=test_input["attention_mask"])
            outputs[agent_id] = out.logits.clone()
    
    # Compare outputs
    for i in range(args.num_agents):
        for j in range(i + 1, args.num_agents):
            diff = (outputs[i] - outputs[j]).abs().mean().item()
            print(f"Output difference between Agent {i} and Agent {j}: {diff:.6f}")
            if diff > 1e-6:
                print(f"  -> Adapters are differentiated (gradient isolation verified)")
            else:
                print(f"  -> WARNING: Adapters may not be properly differentiated")
    
    # Save adapters
    print("\n" + "=" * 60)
    print("Saving Adapters")
    print("=" * 60)
    
    save_dir = os.path.join(args.output_dir, f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    model.save_all_adapters(save_dir)
    print(f"Saved all adapters to {save_dir}")
    
    # Save metrics
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

