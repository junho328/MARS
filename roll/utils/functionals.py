import enum
import traceback
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.utils.kl_controller import AdaptiveKLController
from roll.utils.logging import get_logger


logger = get_logger()


def tensor_to_cpu_visitor(obj, path):
    if torch.is_tensor(obj):
        if not obj.is_cpu:
            obj.data = obj.data.detach().cpu()
        return True
    return False


def tensor_to_cuda_visitor(obj, path):
    if torch.is_tensor(obj):
        if not obj.is_cuda:
            obj.data = obj.data.detach().to(device=torch.device("cuda"))
        return True
    return False


def delete_tensor_grad_visitor(obj, path):
    if torch.is_tensor(obj):
        obj.grad = None
        return True
    return False


def traverse_obj(value, visitor, path=()):
    """
    Recursively traverse all attributes of an object, including nested attributes, to find all Tensors.
    This is useful for inspecting complex nested data structures and applying operations to all tensors.
    
    Args:
        value: Any Python object to traverse (dict, list, tuple, or custom object)
        visitor: Callback function that receives (value, path) and returns True to stop traversal
        path: Tuple tracking the current traversal path (e.g., ('key1', 0, 'attr:name'))
    """
    if visitor(value, path):
        return
    elif isinstance(value, dict):
        for key, value in value.items():
            traverse_obj(value, visitor, path + (str(key),))
    elif isinstance(value, list) or isinstance(value, tuple):
        for index, item in enumerate(value):
            traverse_obj(item, visitor, path + (index,))
    elif hasattr(value, "__dict__"):
        for attr_name in dir(value):
            if not attr_name.startswith("__"):
                try:
                    attr_value = getattr(value, attr_name)
                    traverse_obj(attr_value, visitor, path + (f"attr:{attr_name}",))
                except Exception as e:
                    logger.error(e)
                    continue


def union_two_dict(dict1: Dict, dict2: Dict):
    """Union two dict. Will throw an error if there is an item not the same object with the same key.

    Args:
        dict1:
        dict2:

    Returns:

    """
    for key, val in dict2.items():
        if key in dict1:
            if isinstance(val, dict):
                val = union_two_dict(dict1[key], val)
            else:
                assert dict2[key] == dict1[key], f"{key} in meta_dict1 and meta_dict2 are not the same object"
        dict1[key] = val

    return dict1


def divide_by_chunk_size(
    data: Union[np.ndarray, TensorDict], chunk_sizes: List[int]
) -> List[Union[np.ndarray, TensorDict]]:
    """
    Split numpy array by chunks size
    """
    if not isinstance(data, (np.ndarray, TensorDict)):
        raise TypeError("Input 'array' must be a numpy ndarray or a TensorDict.")

    if not all(isinstance(size, int) and size > 0 for size in chunk_sizes):
        raise ValueError("All chunk sizes must be positive integers.")

    total_size = sum(chunk_sizes)
    if total_size != len(data):
        raise ValueError(f"The sum of chunk_sizes ({total_size}) does not match the size of the array ({len(data)}).")

    split_data = []
    start_index = 0
    for size in chunk_sizes:
        end_index = start_index + size
        split_data.append(data[start_index:end_index])
        start_index = end_index
    return split_data


def append_to_dict(data: Dict, new_data: Dict):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """
        xs_count = xs.numel()
        xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()


def compute_clip_fraction(values: torch.Tensor, clip_max: float, clip_min: float):
    numel = values.numel()
    num_clipped = (values > clip_max).sum().item() + (values < clip_min).sum().item()
    clipfrac = num_clipped / numel if numel > 0 else 0.0
    return clipfrac


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    kl_penalty: str = "kl",
) -> torch.Tensor:
    """
    ref: https://github.com/OpenRLHF/OpenRLHF/blob/494850f50342ed38d5ae76ef45a3207f3523b582/openrlhf/models/utils.py#L7
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html
    """
    if kl_penalty == "kl":
        log_ratio = log_probs - log_probs_base
    elif kl_penalty == "abs":
        log_ratio = (log_probs - log_probs_base).abs()
    elif kl_penalty == "mse":
        log_ratio = 0.5 * (log_probs - log_probs_base).square()
    elif kl_penalty == "k3":
        kl = log_probs_base - log_probs
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        log_ratio = torch.clamp(kld, min=-10, max=10)
    elif kl_penalty == "full":
        log_ratio = F.kl_div(log_probs_base, log_probs, log_target=True, reduction="none").sum(-1)
    else:
        raise NotImplementedError

    if action_mask is not None:
        return log_ratio * action_mask

    return log_ratio


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str,
             weights: Optional[torch.Tensor] = None):
    """
    ref: https://github.com/volcengine/verl/blob/78532923368aeb058f62201489546d013df47710/verl/trainer/ppo/core_algos.py#L370
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "seq-mean-token-sum" is the default behavior
        weights: `torch.Tensor`
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if weights is None:
            weights = torch.ones(loss_mask.shape[0], device=loss_mask.device)
        loss = masked_mean(loss_mat * weights.unsqueeze(-1), loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = masked_mean(loss_mat, loss_mask, dim=-1) # token-sum
        valid_samples = torch.any(loss_mask > 0, dim=1).float()
        if weights is None:
            weights = torch.ones(loss_mask.shape[0], device=loss_mask.device)
        loss = (seq_losses * weights * valid_samples).sum() / (valid_samples.sum() + 1e-8) # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = masked_mean(loss_mat, loss_mask, dim=-1)
        seq_losses = seq_losses / (torch.sum(loss_mask, dim=-1) + 1e-8)  # token-mean
        valid_samples = torch.any(loss_mask > 0, dim=1).float()
        if weights is None:
            weights = torch.ones(loss_mask.shape[0], device=loss_mask.device)
        loss = (seq_losses * weights * valid_samples).sum() / (valid_samples.sum() + 1e-8)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = masked_mean(loss_mat, loss_mask, dim=-1)
        valid_samples = torch.any(loss_mask > 0, dim=1).float()
        if weights is None:
            weights = torch.ones(loss_mask.shape[0], device=loss_mask.device)
        loss = (seq_losses * weights * valid_samples).sum() / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
    if dim is not None:
        mask_sum = mask.sum(axis=dim)
        return torch.where(mask_sum > 0, (tensor * mask).sum(axis=dim) / (mask_sum + 1e-8), torch.zeros_like(mask_sum))
    else:
        return (
            (tensor * mask).sum() / (mask.sum() + 1e-8) if mask.sum() > 0 else torch.tensor(0.0, device=tensor.device)
        )


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def get_eos_mask(response_id: torch.Tensor, eos_token: int = 2, dtype=torch.int64):
    """
    e.g. end of sentence token=1
    response_id: [0, 0, 2, 42, 3, 5, 1, 0, 0]
    eos_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
    """
    eos_mask = response_id.eq(eos_token).long()
    eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
    eos_mask = torch.logical_not(eos_mask).to(dtype)
    return eos_mask


def get_pad_mask(response_id: torch.Tensor, pad_token: int = 0, dtype=torch.int64):
    """
    e.g. pad token=0
    response_id: [1, 2, 2, 42, 3, 5, 1, 0, 0]
    pad_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
    """
    pad_mask = response_id.not_equal(pad_token).to(dtype)
    assert (
        not (pad_mask[:, 0] == 0).logical_and(pad_mask.sum(-1) != 0).any()
    ), f"response_id is not valid: {response_id}, pad_token is {pad_token}"
    return pad_mask


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim).unsqueeze(-1)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim).unsqueeze(-1)
    return mean_centered * var.clamp(min=eps).rsqrt()


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def response_level_masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True):
    """Whiten values with masked values."""
    # Consider the influence of response?
    mean = masked_mean(values, mask, dim=-1)
    var = masked_var(mean, mask)
    mean = mean.mean()
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def reduce_metrics(metrics: dict, reduce_func=np.mean) -> dict:
    for key, val in metrics.items():
        metrics[key] = reduce_func(val)
    return metrics


def pad_to_length(tensor: torch.Tensor, length, pad_value, dim=-1):
    if tensor.size(dim) >= length:
        indices = [slice(None)] * tensor.ndim
        indices[dim] = slice(0, length)
        return tensor[indices]
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
        )


def concatenate_input_and_output(input_ids, output_ids, num_return_sequences):
    batch_size, input_seq_len = input_ids.size()
    _, output_seq_len = output_ids.size()
    repeated_input_ids = (
        input_ids.unsqueeze(1)
        .repeat(1, num_return_sequences, 1)
        .view(batch_size * num_return_sequences, input_seq_len)
    )
    sequences = torch.cat((repeated_input_ids, output_ids), dim=1)
    return sequences


def compute_reinforce_return(token_level_rewards: torch.Tensor, gamma: torch.Tensor, lambd: torch.Tensor):
    """
    Compute discounted return-to-go for REINFORCE/GRPO algorithm.
    
    This implements the classic Monte Carlo return calculation:
        G_t = r_t + gamma * G_{t+1}
    where G_t is the return-to-go at timestep t.
    
    Used by:
    - REINFORCE: Basic policy gradient with full episode returns
    - GRPO (Group Relative Policy Optimization): REINFORCE + group normalization
    
    Args:
        token_level_rewards: Rewards at each token position, shape (batch_size, seq_len)
        gamma: Discount factor (typically 0.99 or 1.0 for undiscounted)
        lambd: Not used in REINFORCE (kept for API compatibility with GAE)
        
    Returns:
        advantages: Return-to-go at each position, shape (batch_size, seq_len)
        returns: Same as advantages for REINFORCE (no baseline subtraction)
    """
    with torch.no_grad():
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]
        cumulative_reward = 0
        for t in reversed(range(gen_len)):
            local_reward = token_level_rewards[:, t] if t < gen_len else 0.0
            cumulative_reward = local_reward + gamma * cumulative_reward
            advantages_reversed.append(cumulative_reward)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages
    return advantages, returns


def compute_microstep_return(
    token_level_rewards: torch.Tensor,
    turn_end_positions: torch.Tensor,
    gamma: float,
    lambd: float = None,
):
    """
    Compute turn-level (microstep) return-to-go for multi-agent GRPO.
    
    This aggregates token-level rewards to turn boundaries and computes returns
    at turn granularity, providing cleaner credit assignment than token-level computation.
    
    Algorithm:
    1. Aggregate token-level rewards within each turn to get turn_level_rewards
    2. Compute return-to-go at turn boundaries: G_turn_t = r_turn_t + gamma * G_turn_{t+1}
    3. Broadcast turn-level returns back to all tokens within that turn
    
    Example:
        Token rewards:    [0, 0, 5, 0, 0, 0, 10, 0]
        Turn boundaries:  [0, 0, 1, 0, 0, 0, 1,  0]  (1 marks turn end)
        Turn rewards:     [5, 10]
        Turn returns:     [5 + gamma*10, 10]
        Broadcasted:      [5+10g, 5+10g, 5+10g, 10, 10, 10, 10, 10]
    
    Args:
        token_level_rewards: Rewards at each token, shape (batch_size, seq_len)
        turn_end_positions: Boolean mask marking turn boundaries, shape (batch_size, seq_len)
        gamma: Discount factor (typically 1.0 for undiscounted multi-agent scenarios)
        lambd: Not used (kept for API compatibility)
        
    Returns:
        advantages: Turn-level return-to-go broadcasted to token level, shape (batch_size, seq_len)
        returns: Same as advantages (no baseline subtraction)
    """
    with torch.no_grad():
        batch_size, seq_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)  # (batch_size, seq_len)
        
        for batch_idx in range(batch_size):
            # Find turn end positions for this trajectory
            turn_ends = torch.where(turn_end_positions[batch_idx])[0].cpu().numpy()
            
            if len(turn_ends) == 0:
                # No turns identified, fall back to token-level computation
                advantages_reversed = []
                cumulative_reward = 0
                for t in reversed(range(seq_len)):
                    local_reward = token_level_rewards[batch_idx, t]
                    cumulative_reward = local_reward + gamma * cumulative_reward
                    advantages_reversed.append(cumulative_reward)
                advantages[batch_idx] = torch.stack(advantages_reversed[::-1])
                continue
            
            # Step 1: Aggregate rewards at turn level
            turn_rewards = []
            prev_end = -1
            for turn_end in turn_ends:
                # Sum rewards from prev_end+1 to turn_end (inclusive)
                turn_reward = token_level_rewards[batch_idx, prev_end + 1 : turn_end + 1].sum()
                turn_rewards.append(turn_reward)
                prev_end = turn_end
            
            # Step 2: Compute return-to-go at turn level
            num_turns = len(turn_rewards)
            turn_returns = torch.zeros(num_turns, device=token_level_rewards.device)
            cumulative_return = 0.0
            for turn_idx in reversed(range(num_turns)):
                cumulative_return = turn_rewards[turn_idx] + gamma * cumulative_return
                turn_returns[turn_idx] = cumulative_return
            
            # Step 3: Broadcast turn-level returns back to tokens
            prev_end = -1
            for turn_idx, turn_end in enumerate(turn_ends):
                start_idx = prev_end + 1
                end_idx = turn_end + 1
                # Assign the turn's return to all tokens in this turn
                advantages[batch_idx, start_idx:end_idx] = turn_returns[turn_idx]
                prev_end = turn_end
        
        returns = advantages
    return advantages, returns


def compute_cooperative_microstep_return(
    data: "DataProto",
    token_level_rewards: torch.Tensor,
    turn_end_positions: torch.Tensor,
    gamma: float,
    lambd: float = None,
):
    """
    Compute turn-level return-to-go with turn-position-wise normalization ("Ours" algorithm).
    
    Implements Equation (10) from the paper:
        A_k^g = (R_k^g - mean(R_k)) / std(R_k)
    
    where R_k = {R_k^g1, R_k^g2, ...} is the set of returns at turn position k across all games.
    
    Key principle: Normalization only happens when |R_k| > 1. 
    If a turn position has only 1 return, we cannot normalize (set advantage to 0).
    
    Example (3 games, 4 turns each):
        Rewards: {(1,2,3,4), (5,6,7,8), (9,10,11,12)}
        Returns: {(10,9,7,4), (26,21,15,8), (42,33,23,12)}
        
        Microstep returns grouped by turn k:
        R_1 = (10, 26, 42) → normalize together (3 values)
        R_2 = (9, 21, 33) → normalize together (3 values)
        R_3 = (7, 15, 23) → normalize together (3 values)
        R_4 = (4, 8, 12) → normalize together (3 values)
    
    Args:
        data: DataProto containing traj_group_id for trajectory pairing
        token_level_rewards: Rewards at each token, shape (batch_size, seq_len)
        turn_end_positions: Boolean mask marking turn boundaries, shape (batch_size, seq_len)
        gamma: Discount factor (typically 1.0)
        lambd: Not used (kept for API compatibility)
        
    Returns:
        advantages: Turn-position-normalized returns, shape (batch_size, seq_len)
        returns: Same as advantages
    """
    with torch.no_grad():
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device
        dtype = token_level_rewards.dtype
        
        # Validate input tensors - replace NaN/Inf with zeros
        if torch.isnan(token_level_rewards).any() or torch.isinf(token_level_rewards).any():
            token_level_rewards = torch.where(
                torch.isfinite(token_level_rewards), 
                token_level_rewards, 
                torch.zeros_like(token_level_rewards)
            )
        
        # Initialize advantages to zeros
        advantages = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        
        # Get trajectory IDs for pairing
        traj_ids = data.non_tensor_batch.get("traj_group_id", [])
        if len(traj_ids) == 0:
            traj_ids = data.non_tensor_batch.get("group_ids", [])
        
        if len(traj_ids) == 0:
            # No trajectory IDs - fall back to reinforce
            logger.warning("[compute_cooperative_microstep_return] No trajectory IDs found. "
                          "Falling back to reinforce.")
            return compute_reinforce_return(token_level_rewards, gamma, lambd)
        
        # STEP 1: Parse trajectory IDs and group by game
        # game_map: game_id -> {player_id -> batch_idx}
        game_map = {}
        
        for batch_idx in range(batch_size):
            if batch_idx >= len(traj_ids):
                continue
                
            traj_id = traj_ids[batch_idx]
            if hasattr(traj_id, 'item'):
                traj_id = traj_id.item()
            elif isinstance(traj_id, np.ndarray):
                traj_id = str(traj_id.flat[0]) if traj_id.size > 0 else str(traj_id)
            traj_id = str(traj_id)
            
            # Parse player suffix: "0_1_123_p0" -> game_id="0_1_123", player_id=0
            if "_p" in traj_id:
                parts = traj_id.rsplit("_p", 1)
                game_id = parts[0]
                player_id = int(parts[1]) if parts[1].isdigit() else 0
            else:
                # Orphaned trajectory - treat as its own single-player game
                game_id = f"orphan_{batch_idx}"
                player_id = 0
            
            if game_id not in game_map:
                game_map[game_id] = {}
            game_map[game_id][player_id] = batch_idx
        
        # STEP 2: For each game, extract turns and compute interleaved returns
        # turn_position_data[turn_k] = list of dicts with batch_idx, return, start, end
        turn_position_data = {}
        processed_indices = set()
        
        for game_id, players in game_map.items():
            # Extract turns for each player in this game
            all_player_turns = {}
            for player_id, batch_idx in players.items():
                turns = extract_turns(token_level_rewards[batch_idx], turn_end_positions[batch_idx])
                all_player_turns[player_id] = {'batch_idx': batch_idx, 'turns': turns}
            
            # Get max turns across players in this game
            max_player_turns = max(len(p['turns']) for p in all_player_turns.values()) if all_player_turns else 0
            
            if max_player_turns == 0:
                # No turns detected for this game - skip (advantages stay 0)
                for player_id, batch_idx in players.items():
                    processed_indices.add(batch_idx)  # Mark as processed (with 0 advantage)
                continue
            
            # Build interleaved sequence: P0_turn0, P1_turn0, P0_turn1, P1_turn1, ...
            interleaved = []
            for turn_idx in range(max_player_turns):
                for player_id in sorted(all_player_turns.keys()):
                    player_data = all_player_turns[player_id]
                    if turn_idx < len(player_data['turns']):
                        interleaved.append((
                            player_id, 
                            turn_idx, 
                            player_data['batch_idx'], 
                            player_data['turns'][turn_idx]
                        ))
            
            # Compute return-to-go for interleaved sequence
            # R_k = r_k + gamma * r_{k+1} + gamma^2 * r_{k+2} + ...
            rewards = []
            for t in interleaved:
                r = t[3]['reward']
                rewards.append(r.item() if hasattr(r, 'item') else float(r))
            
            returns_list = []
            cumulative = 0.0
            for r in reversed(rewards):
                cumulative = r + gamma * cumulative
                returns_list.insert(0, cumulative)
            
            # Store by global turn position k
            for global_k, (player_id, turn_idx, batch_idx, turn_info) in enumerate(interleaved):
                if global_k not in turn_position_data:
                    turn_position_data[global_k] = []
                turn_position_data[global_k].append({
                    'batch_idx': batch_idx,
                    'return': returns_list[global_k],
                    'start': turn_info['start'],
                    'end': turn_info['end'],
                })
                processed_indices.add(batch_idx)
        
        # STEP 3: Normalize by turn position k across all games ("Ours" - Equation 10)
        # A_k^g = (R_k^g - mean(R_k)) / std(R_k)
        # If |R_k| == 1, cannot normalize → use raw return to preserve learning signal
        
        for turn_k, pos_data in turn_position_data.items():
            returns_at_k = torch.tensor([d['return'] for d in pos_data], device=device, dtype=dtype)
            
            if len(returns_at_k) > 1:
                # Multiple games have this turn position → can normalize
                mean_k = returns_at_k.mean()
                std_k = returns_at_k.std()
                
                if std_k > 1e-8:
                    # Normal case: A_k^g = (R_k^g - mean(R_k)) / std(R_k)
                    normalized_k = (returns_at_k - mean_k) / std_k
                else:
                    # All returns identical (std=0) → mean-center only (results in all zeros)
                    normalized_k = returns_at_k - mean_k
                
                # Clamp to prevent extreme values
                normalized_k = torch.clamp(normalized_k, min=-10.0, max=10.0)
            else:
                # Only 1 game has this turn position → cannot normalize
                # Use raw return to preserve learning signal
                normalized_k = torch.clamp(returns_at_k, min=-10.0, max=10.0)
            
            # Broadcast normalized returns to token level
            for i, data_point in enumerate(pos_data):
                batch_idx = data_point['batch_idx']
                start = data_point['start']
                end = min(data_point['end'], seq_len)
                
                if batch_idx < batch_size and start < seq_len and end <= seq_len:
                    norm_val = normalized_k[i].item()
                    if np.isfinite(norm_val):
                        advantages[batch_idx, start:end] = norm_val
        
        # STEP 4: Final safety checks
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            logger.error("[compute_cooperative_microstep_return] NaN/Inf in advantages! Replacing with zeros.")
            advantages = torch.where(torch.isfinite(advantages), advantages, torch.zeros_like(advantages))
        
        # Log summary
        num_turn_positions = len(turn_position_data)
        sizes = [len(pos_data) for pos_data in turn_position_data.values()]
        if sizes:
            logger.debug(f"[compute_cooperative_microstep_return] {num_turn_positions} turn positions, "
                        f"group sizes: min={min(sizes)}, max={max(sizes)}, "
                        f"positions with >1 game: {sum(1 for s in sizes if s > 1)}")
        
        # Ensure contiguous memory
        advantages = advantages.contiguous()
        returns = advantages.clone()
        
        return advantages, returns


def extract_turns(token_rewards: torch.Tensor, turn_end_mask: torch.Tensor):
    """
    Helper function to extract turn information from token-level data.
    
    Args:
        token_rewards: Rewards for a single trajectory, shape (seq_len,)
        turn_end_mask: Turn boundary mask, shape (seq_len,)
        
    Returns:
        List of dicts with 'start', 'end', and 'reward' for each turn
        Each dict: {'start': int, 'end': int, 'reward': scalar tensor}
    """
    seq_len = len(token_rewards)
    turn_ends = torch.where(turn_end_mask)[0].cpu().numpy()
    turns = []
    prev_end = -1
    
    for turn_end in turn_ends:
        start_idx = prev_end + 1
        end_idx = min(turn_end + 1, seq_len)  # Ensure we don't exceed sequence length
        
        if start_idx >= seq_len or start_idx >= end_idx:
            continue  # Skip invalid ranges
            
        turn_reward = token_rewards[start_idx:end_idx].sum()
        turns.append({
            'start': start_idx,
            'end': end_idx,
            'reward': turn_reward
        })
        prev_end = turn_end
    
    return turns





def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor, values: torch.Tensor, gamma: torch.Tensor, lambd: torch.Tensor
):
    """
    Compute Generalized Advantage Estimation (GAE) for more stable policy gradients.
    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py
    
    GAE uses TD(λ) to balance bias-variance tradeoff:
        δ_t = r_t + gamma * V(s_{t+1}) - V(s_t)  [TD error]
        A_t^{GAE} = Σ_{l=0}^∞ (gamma * lambda)^l * δ_{t+l}
    
    Benefits:
    - Reduces variance compared to Monte Carlo (lambd=1.0)
    - Less biased than 1-step TD (lambd=0.0)
    - Typical lambd=0.95 provides good balance

    Args:
        token_level_rewards: Rewards at each token, shape (batch_size, seq_len)
        values: Value function estimates V(s_t), shape (batch_size, seq_len+1)
                Last value is bootstrap value for final state
        gamma: Discount factor for future rewards (typically 0.99)
        lambd: GAE lambda parameter controlling bias-variance (typically 0.95)
        
    Returns:
        advantages: GAE advantages A_t^{GAE}, shape (batch_size, seq_len)
        returns: Target values (advantages + baseline), shape (batch_size, seq_len)
    
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        gamma: `(float)`
            discounted factor used in RL
        lambd: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values

    return advantages, returns


def expand_to_token_level(data: "DataProto"):
    response_level_rewards = data.batch["response_level_rewards"].clone().detach()
    batch_size = data.batch.batch_size[0]
    # expand as token_level_rewards
    attention_mask = data.batch["attention_mask"]
    position_ids = data.batch["position_ids"]
    if position_ids.dim() == 3:
        # qwen2vl, (bsz, 3, seqlen), 0/1/2 is same for text, while values of
        # position_ids for text cannot stand for index of tokens, thus use the
        # right padding attention_mask to calculate eos index or `argmax` rather
        # than `max` of position_ids to calculate eos index
        position_ids = position_ids[:, 0]
    eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
    token_level_rewards = torch.zeros_like(attention_mask, dtype=response_level_rewards.dtype)  # (bsz, seqlen)

    token_level_rewards[torch.arange(batch_size), eos_mask_idx] = response_level_rewards

    # select the response part
    token_level_rewards = token_level_rewards[:, 1:]

    return token_level_rewards


def batch_reward_norm(response_level_rewards: torch.Tensor, div_std=True):
    batch_mean = response_level_rewards.mean()
    if div_std:
        normalized_rewards = (response_level_rewards - batch_mean) / (response_level_rewards.std() + 1e-6)
    else:
        normalized_rewards = response_level_rewards - batch_mean
    return normalized_rewards


def normalize_unique_values(tensor: torch.Tensor, mode="mean") -> torch.Tensor:
    """
    Normalize all unique values in a tensor while preserving duplicate structure.
    This is useful for advantage normalization where we want to normalize the unique advantage values
    rather than all individual tokens, which helps with stability in multi-turn scenarios.
    
    Example: 
        Input:  [1, 1, 1, 2, 2, 3, 3, 3, 3]
        Unique: [1, 2, 3] -> normalized to [-1, 0, 1]
        Output: [-1, -1, -1, 0, 0, 1, 1, 1, 1]
    
    Args:
        tensor: Input tensor of any shape (typically (batch_size, seq_len))
        mode: Normalization mode
            - "mean": Subtract mean only (zero-center)
            - "mean_std": Z-score normalization (subtract mean, divide by std)
        
    Returns:
        Normalized tensor maintaining original shape and duplicate structure
    """
    with torch.no_grad():
        # Get all unique values
        unique_values = torch.unique(tensor)
        
        # If only one unique value, return zero tensor
        if len(unique_values) == 1:
            return torch.zeros_like(tensor)
        
        # Normalize unique values: subtract mean, divide by std
        unique_mean = unique_values.mean()
        unique_std = unique_values.std()
        
        if unique_std == 0:
            # If std is 0, all values are identical; return zero tensor
            return torch.zeros_like(tensor)
        
        if mode == "mean_std":
            normalized_unique = (unique_values - unique_mean) / (unique_std + 1e-6)
        elif mode == "mean":
            normalized_unique = unique_values - unique_mean
        else:
            raise ValueError(f"Invalid normalization mode: {mode}")
        
        # Create mapping dictionary: original value -> normalized value
        value_mapping = {}
        for i, original_val in enumerate(unique_values):
            value_mapping[original_val.item()] = normalized_unique[i].item()
        
        # Create result tensor to hold normalized values
        result = torch.zeros_like(tensor)
        
        # Replace each unique value with its normalized counterpart
        for original_val, normalized_val in value_mapping.items():
            mask = (tensor == original_val)
            result[mask] = normalized_val
            
        return result


def normalize_unique_values_by_player(tensor: torch.Tensor, data: "DataProto", mode="mean") -> torch.Tensor:
    """
    Normalize all unique values in a tensor, processing each player separately.
    This follows the implementation pattern of reward_normalize_by_player to ensure
    fair normalization in self-play scenarios where Player 0 and Player 1 may have
    different reward distributions.
    
    Args:
        tensor: Input tensor of any shape (typically advantages or rewards)
        data: DataProto containing group_ids with player information (e.g., "env_0_p0", "env_1_p1")
        mode: Normalization mode
            - "mean": Subtract mean only
            - "mean_std": Z-score normalization
        
    Returns:
        Normalized tensor maintaining original shape, with per-player normalization applied
    """
    # Extract player information from group_ids
    group_ids = data.non_tensor_batch.get("group_ids", [])
    if len(group_ids) == 0:
        # If no group_ids available, fallback to standard normalization
        return normalize_unique_values(tensor, mode=mode)
    
    # Identify indices for each player (player 0 and player 1)
    player_0_indices = []
    player_1_indices = []
    
    for i, group_id in enumerate(group_ids):
        if isinstance(group_id, str) and "_p" in group_id:
            player_id = group_id.split("_p")[-1]
            if player_id == "0":
                player_0_indices.append(i)
            elif player_id == "1":
                player_1_indices.append(i)
        else:
            # If no player marker found, default to player 0
            player_0_indices.append(i)
    
    # If no clear player separation found, fallback to standard normalization
    if len(player_0_indices) == 0 and len(player_1_indices) == 0:
        return normalize_unique_values(tensor, mode=mode)
    
    # Create result tensor to hold normalized values
    normalized_tensor = tensor.clone()
    
    # Normalize player 0's tensor (if exists)
    if len(player_0_indices) > 0:
        player_0_indices_tensor = torch.tensor(player_0_indices, dtype=torch.long)
        player_0_tensor = tensor[player_0_indices_tensor]
        
        player_0_normalized = normalize_unique_values(player_0_tensor, mode=mode)
        normalized_tensor[player_0_indices_tensor] = player_0_normalized
    
    # Normalize player 1's tensor (if exists)
    if len(player_1_indices) > 0:
        player_1_indices_tensor = torch.tensor(player_1_indices, dtype=torch.long)
        player_1_tensor = tensor[player_1_indices_tensor]
        
        player_1_normalized = normalize_unique_values(player_1_tensor, mode=mode)
        normalized_tensor[player_1_indices_tensor] = player_1_normalized
    
    return normalized_tensor


def group_reward_norm(data: "DataProto", n_sample=-1, div_std=True, div_std_global=False):
    assert n_sample > 1, "n_sample must > 1"
    response_level_rewards = data.batch["response_level_rewards"].clone().detach()
    reshape_reward = response_level_rewards.reshape(*response_level_rewards.size()[:-1], -1, n_sample)
    reshape_reward = reshape_reward - reshape_reward.mean(dim=-1, keepdim=True)
    if div_std:
        if not div_std_global:
            reshape_reward = reshape_reward / (torch.std(reshape_reward, dim=-1, keepdim=True) + 1e-6)
        else:
            reshape_reward = reshape_reward / (torch.std(reshape_reward) + 1e-6)
    data.batch["response_level_rewards"] = reshape_reward.reshape(*response_level_rewards.size())
    return data


def difficulty_mask(data: "DataProto", n_sample=-1, low_threshold=0.1, high_threshold=0.95):
    if n_sample > 1:
        scores = data.batch["scores"].clone().detach()
        reshape_score = scores.reshape(*scores.size()[:-1], -1, n_sample)
        reshape_score_mean = reshape_score.mean(dim=-1, keepdim=True).expand_as(reshape_score).reshape(*scores.size())
        data.batch["difficulty_mask"] = (reshape_score_mean > low_threshold) * (reshape_score_mean < high_threshold)
    else:
        data.batch["difficulty_mask"] = torch.ones_like(data.batch["scores"])
    return data


@torch.no_grad()
def compute_token_reward(data: "DataProto", pipeline_config: RLVRConfig, kl_ctrl: AdaptiveKLController):
    token_level_rewards = expand_to_token_level(data)
    beta = 0
    kld = compute_approx_kl(
        log_probs=data.batch["old_log_probs"],
        log_probs_base=data.batch["ref_log_probs"],
        action_mask=data.batch["response_mask"][:, 1:],
        kl_penalty=pipeline_config.kl_penalty,
    )
    # Whether to add token level kl
    if pipeline_config.add_token_level_kl and "ref_log_probs" in data.batch.keys():
        beta = kl_ctrl.value
        token_level_rewards = token_level_rewards - beta * kld

    current_kl = masked_mean(kld, mask=data.batch["response_mask"][:, 1:], dim=-1)
    current_kl = torch.mean(current_kl, dim=0).item()

    kl_ctrl.update(current=current_kl, n_steps=data.batch.batch_size[0])
    if "token_level_rewards" in data.batch.keys():
        data.rename(old_keys="token_level_rewards", new_keys="token_level_scores")
    metrics = {"critic/kl": current_kl, "critic/kl_coef": beta}

    if pipeline_config.reward_clip:
        reward_clip_frac = compute_clip_fraction(
            values=token_level_rewards, clip_max=pipeline_config.reward_clip, clip_min=-pipeline_config.reward_clip
        )
        metrics["critic/token_reward_clip_frac"] = reward_clip_frac
        token_level_rewards = torch.clamp(
            token_level_rewards, min=-pipeline_config.reward_clip, max=pipeline_config.reward_clip
        )

    data.batch["token_level_rewards"] = token_level_rewards
    return data, metrics


@torch.no_grad()
def reward_postprocess(data: "DataProto", pipeline_config: RLVRConfig, running_ctrl):
    response_level_rewards = data.batch["response_level_rewards"].clone().detach()
    response_level_metrics = {"critic/reward_clip_frac": 0.0}
    # Process rewards: can choose different normalization methods
    # Use group-based normalization (grouped by prompt)
    if pipeline_config.adv_estimator == "grpo" or pipeline_config.reward_norm == "group":
        if pipeline_config.reward_shift:
            data = group_reward_norm(
                data,
                n_sample=pipeline_config.actor_infer.generating_args.num_return_sequences,
                div_std=False,
            )
        else:
            data = group_reward_norm(
                data,
                n_sample=pipeline_config.actor_infer.generating_args.num_return_sequences,
                div_std=True,
            )
        response_level_rewards = data.batch["response_level_rewards"].clone().detach()

    # Use batch-based normalization (entire batch)
    elif pipeline_config.reward_norm == "batch":
        if hasattr(pipeline_config, "reward_shift") and pipeline_config.reward_shift:
            response_level_rewards = batch_reward_norm(response_level_rewards, div_std=False)
        else:
            response_level_rewards = batch_reward_norm(response_level_rewards, div_std=True)

    # Use running statistics for normalization
    elif pipeline_config.reward_norm == "running":
        running = running_ctrl["domain"]
        running.update(response_level_rewards)
        mean = running.mean
        std = running.std + torch.finfo(response_level_rewards.dtype).eps
        if pipeline_config.reward_shift:
            response_level_rewards = response_level_rewards - mean
        elif pipeline_config.reward_scale:
            response_level_rewards = response_level_rewards / std
        else:
            response_level_rewards = (response_level_rewards - mean) / std

    # Clip rewards
    if pipeline_config.reward_clip:
        reward_clip_frac = compute_clip_fraction(
            values=response_level_rewards, clip_max=pipeline_config.reward_clip, clip_min=-pipeline_config.reward_clip
        )
        response_level_rewards = torch.clamp(
            response_level_rewards, min=-pipeline_config.reward_clip, max=pipeline_config.reward_clip
        )

        response_level_metrics = {"critic/reward_clip_frac": reward_clip_frac}

    data.batch["response_level_rewards"] = response_level_rewards
    return data, response_level_metrics


@torch.no_grad()
def get_sample_level_mask(data: "DataProto", pipeline_config: RLVRConfig):
    batch_size = data.batch["response_mask"].size(0)
    mask_metrics = {}

    # Mask-related strategies to filter out low-quality or problematic samples
    data.batch["origin_response_mask"] = data.batch["response_mask"].clone()
    response_mask = data.batch["response_mask"][:, 1:].clone()
    true_response_length = response_mask.sum(-1).float()
    max_response_length = data.batch["responses"].shape[-1]

    final_sample_mask = torch.ones(batch_size, device=response_mask.device)

    # 1. max_len_mask: Filter out samples that exceeded maximum length during generation
    if pipeline_config.max_len_mask:
        max_len_mask = (max_response_length != true_response_length).float()
        final_sample_mask = final_sample_mask * max_len_mask
        mask_metrics["actor/max_len_mask_ratio"] = max_len_mask.mean().item()
    else:
        mask_metrics["actor/max_len_mask_ratio"] = 1.0

    # 2. difficulty_mask: difficulty-based filtering
    if pipeline_config.difficulty_mask:
        data = difficulty_mask(
            data,
            n_sample=pipeline_config.actor_infer.generating_args.num_return_sequences,
            low_threshold=pipeline_config.difficulty_low_threshold,
            high_threshold=pipeline_config.difficulty_high_threshold,
        )
        if "difficulty_mask" in data.batch:
            difficulty_mask_tensor = data.batch["difficulty_mask"].float()
            final_sample_mask = final_sample_mask * difficulty_mask_tensor
            mask_metrics["actor/difficulty_mask_ratio"] = difficulty_mask_tensor.mean().item()
        else:
            mask_metrics["actor/difficulty_mask_ratio"] = 1.0
    else:
        mask_metrics["actor/difficulty_mask_ratio"] = 1.0

    # 3. error_max_len_clip: filtering based on errors and length
    if pipeline_config.error_max_len_clip:
        scores = data.batch["scores"]
        error_len_mask = ((scores == 0) & (true_response_length < pipeline_config.error_max_len_threshold)) | (
            scores == 1
        )
        error_len_mask = error_len_mask.float()
        final_sample_mask = final_sample_mask * error_len_mask
        mask_metrics["actor/error_len_mask_ratio"] = error_len_mask.mean().item()
    else:
        mask_metrics["actor/error_len_mask_ratio"] = 1.0

    expanded_sample_mask = final_sample_mask.unsqueeze(-1).expand_as(response_mask)
    final_response_mask = response_mask * expanded_sample_mask
    if final_response_mask.sum() == 0:
        final_response_mask = data.batch["response_mask"][:, 1:].clone()
    mask_metrics["actor/final_mask_ratio"] = final_sample_mask.mean().item()
    mask_metrics["actor/samples_used"] = final_sample_mask.sum().item()
    mask_metrics["actor/samples_total"] = float(batch_size)

    data.batch["final_response_mask"] = final_response_mask
    return data, mask_metrics

# def score_normalize(x, rn_cfg, running_ctrl=None, mask=None) -> torch.Tensor:
#     grouping, method = rn_cfg.grouping, rn_cfg.method
#     if method == "identity" or method == "none":
#         return x
#     else:
#         if mask is None:
#             mean = x.mean()
#             std = x.std()
#         else:
#             mean = masked_mean(x, mask)
#             var = masked_var(x, mask)
#             std = torch.sqrt(var)

#     if method == "mean":
#         x_norm = (x - mean)
#     elif method == "mean_std":
#         x_norm = (
#             (x - mean) / (std + 1e-6)
#             if std.abs().max() > 1e-6
#             else torch.zeros_like(x)
#         )  # stable to bf16 than x.std()
#     elif method == "asym_clip":
#         x_norm = (
#             (x - mean) / (std + 1e-6)
#             if std.abs().max() > 1e-6
#             else torch.zeros_like(x)
#         ).clamp(min=-1, max=3)
#     elif method == "running" and running_ctrl is not None:
#         running_ctrl.update(x)
#         mean = running_ctrl.mean
#         std = running_ctrl.std
#         x_norm = (
#             (x - mean) / (std + 1e-6)
#             if std.abs().max() > 1e-6
#             else torch.zeros_like(x)
#         )
#     else:
#         raise ValueError(f"Invalid normalization method: {method}")
#     if mask is not None:
#         x_norm = x_norm * mask
#     return x_norm

def score_normalize(x, rn_cfg, running_ctrl=None, mask=None) -> torch.Tensor:
    grouping, method = rn_cfg.grouping, rn_cfg.method
    if method == "identity" or method == "none":
        return x
    
    # mean 계산 (모든 method에서 필요)
    if mask is None:
        mean = x.mean()
    else:
        mean = masked_mean(x, mask)
    
    if method == "mean":
        x_norm = (x - mean)
    elif method == "mean_std":
        # std가 필요한 경우에만 계산
        if mask is None:
            std = x.std()
        else:
            var = masked_var(x, mask)
            std = torch.sqrt(var)
        x_norm = (
            (x - mean) / (std + 1e-6)
            if std.abs().max() > 1e-6
            else torch.zeros_like(x)
        )  # stable to bf16 than x.std()
    elif method == "asym_clip":
        # std가 필요한 경우에만 계산
        if mask is None:
            std = x.std()
        else:
            var = masked_var(x, mask)
            std = torch.sqrt(var)
        x_norm = (
            (x - mean) / (std + 1e-6)
            if std.abs().max() > 1e-6
            else torch.zeros_like(x)
        ).clamp(min=-1, max=3)
    elif method == "running" and running_ctrl is not None:
        running_ctrl.update(x)
        mean = running_ctrl.mean
        std = running_ctrl.std
        x_norm = (
            (x - mean) / (std + 1e-6)
            if std.abs().max() > 1e-6
            else torch.zeros_like(x)
        )
    else:
        raise ValueError(f"Invalid normalization method: {method}")
    if mask is not None:
        x_norm = x_norm * mask
    return x_norm


@torch.no_grad()
def reward_normalize_by_player(data: "DataProto", rewards: torch.Tensor, rn_cfg, running_ctrl=None, mask=None, use_turn_level=False):
    """
    Normalize rewards separately for each player in self-play scenarios.
    
    When use_turn_level=True, this function operates on turn-level statistics for microstep estimators,
    normalizing rewards after turn-aggregation but before return-to-go computation.
    
    Args:
        data: DataProto containing trajectory data with group_ids
        rewards: reward tensor to be normalized
        rn_cfg: reward normalization configuration
        running_ctrl: running normalization controller(s) - can be single controller or dict with 'player_0'/'player_1' keys
        mask: optional mask for the rewards (turn_end_positions for turn-level normalization)
        use_turn_level: if True, normalize at turn granularity instead of token level
        
    Returns:
        torch.Tensor: normalized rewards with same shape as input
    """
    # import pdb; pdb.set_trace()
    # Extract player information from group_ids
    group_ids = data.non_tensor_batch.get("group_ids", [])
    if len(group_ids) == 0:
        # Fall back to regular normalization if no group_ids available
        return score_normalize(rewards, rn_cfg=rn_cfg, running_ctrl=running_ctrl, mask=mask)
    
    # Identify player indices
    player_0_indices = []
    player_1_indices = []
    
    for i, group_id in enumerate(group_ids):
        if isinstance(group_id, str) and "_p" in group_id:
            player_id = group_id.split("_p")[-1]
            if player_id == "0":
                player_0_indices.append(i)
            elif player_id == "1":
                player_1_indices.append(i)
        else:
            # If no player info, assign to player 0 by default
            player_0_indices.append(i)
    
    # If we don't find any clear player separation, fall back to regular normalization
    if len(player_0_indices) == 0 and len(player_1_indices) == 0:
        return score_normalize(rewards, rn_cfg=rn_cfg, running_ctrl=running_ctrl, mask=mask)
    
    # Handle running_ctrl - can be single controller or dict with player-specific controllers
    if isinstance(running_ctrl, dict):
        player_0_ctrl = running_ctrl.get("player_0", None)
        player_1_ctrl = running_ctrl.get("player_1", None)
    else:
        # Use the same controller for both players if not separated
        player_0_ctrl = running_ctrl
        player_1_ctrl = running_ctrl
    
    # Create a copy of rewards to modify
    normalized_rewards = rewards.clone()
    
    # For turn-level normalization, we need to extract turn-level rewards
    if use_turn_level and mask is not None:
        # mask is turn_end_positions in this case
        turn_end_positions = mask
        
        # Normalize player 0's rewards at turn level
        if len(player_0_indices) > 0:
            player_0_indices_tensor = torch.tensor(player_0_indices, dtype=torch.long)
            for idx in player_0_indices_tensor:
                turn_ends = torch.where(turn_end_positions[idx])[0]
                if len(turn_ends) == 0:
                    continue
                
                # Extract turn rewards
                turn_rewards = []
                prev_end = -1
                for turn_end in turn_ends:
                    turn_reward = rewards[idx, prev_end + 1 : turn_end + 1].sum()
                    turn_rewards.append(turn_reward)
                    prev_end = turn_end
                
                if len(turn_rewards) > 0:
                    turn_rewards_tensor = torch.stack(turn_rewards)
                    # Normalize at turn level
                    normalized_turn_rewards = score_normalize(
                        turn_rewards_tensor,
                        rn_cfg=rn_cfg,
                        running_ctrl=player_0_ctrl,
                        mask=None
                    )
                    
                    # Broadcast back to tokens
                    prev_end = -1
                    for turn_idx, turn_end in enumerate(turn_ends):
                        normalized_rewards[idx, prev_end + 1 : turn_end + 1] = normalized_turn_rewards[turn_idx]
                        prev_end = turn_end
        
        # Normalize player 1's rewards at turn level
        if len(player_1_indices) > 0:
            player_1_indices_tensor = torch.tensor(player_1_indices, dtype=torch.long)
            for idx in player_1_indices_tensor:
                turn_ends = torch.where(turn_end_positions[idx])[0]
                if len(turn_ends) == 0:
                    continue
                
                # Extract turn rewards
                turn_rewards = []
                prev_end = -1
                for turn_end in turn_ends:
                    turn_reward = rewards[idx, prev_end + 1 : turn_end + 1].sum()
                    turn_rewards.append(turn_reward)
                    prev_end = turn_end
                
                if len(turn_rewards) > 0:
                    turn_rewards_tensor = torch.stack(turn_rewards)
                    # Normalize at turn level
                    normalized_turn_rewards = score_normalize(
                        turn_rewards_tensor,
                        rn_cfg=rn_cfg,
                        running_ctrl=player_1_ctrl,
                        mask=None
                    )
                    
                    # Broadcast back to tokens
                    prev_end = -1
                    for turn_idx, turn_end in enumerate(turn_ends):
                        normalized_rewards[idx, prev_end + 1 : turn_end + 1] = normalized_turn_rewards[turn_idx]
                        prev_end = turn_end
        
        return normalized_rewards
    
    # Standard token-level normalization
    # Normalize player 0's rewards if any exist
    if len(player_0_indices) > 0:
        player_0_indices_tensor = torch.tensor(player_0_indices, dtype=torch.long)
        player_0_rewards = rewards[player_0_indices_tensor]
        player_0_mask = mask[player_0_indices_tensor] if mask is not None else None
        
        player_0_normalized = score_normalize(
            player_0_rewards, 
            rn_cfg=rn_cfg, 
            running_ctrl=player_0_ctrl, 
            mask=player_0_mask
        )
        normalized_rewards[player_0_indices_tensor] = player_0_normalized
    
    # Normalize player 1's rewards if any exist
    if len(player_1_indices) > 0:
        player_1_indices_tensor = torch.tensor(player_1_indices, dtype=torch.long)
        player_1_rewards = rewards[player_1_indices_tensor]
        player_1_mask = mask[player_1_indices_tensor] if mask is not None else None
        
        player_1_normalized = score_normalize(
            player_1_rewards, 
            rn_cfg=rn_cfg, 
            running_ctrl=player_1_ctrl, 
            mask=player_1_mask
        )
        normalized_rewards[player_1_indices_tensor] = player_1_normalized
    
    return normalized_rewards


@torch.no_grad()
def reward_postprocess_agentic(data: "DataProto", pipeline_config: AgenticConfig, running_ctrl=None, kl_ctrl=None):
    # 0. get rewards (process token_level_rewards directly if use_turn_scores is True)
    if pipeline_config.use_turn_scores:
        rewards = data.batch["token_level_rewards"].clone().detach()
        # mask = torch.tensor(rewards != 0, dtype=rewards.dtype)
        mask = data.batch["turn_end_positions"]
    else:
        rewards = data.batch["response_level_rewards"].clone().detach()
        mask = None

    metrics = {"critic/reward_clip_frac": 0.0}

    # Determine if we should use turn-level normalization (for microstep_cooperative estimator)
    use_turn_level_norm = pipeline_config.adv_estimator == "microstep_cooperative"

    # 1. normalize (identity/mean/mean_std/asym_clip/running)
    # IMPORTANT: microstep_cooperative performs turn-position-wise normalization internally!
    # Enforce separate_norm_for_selfplay=False for microstep_cooperative
    if pipeline_config.adv_estimator == "microstep_cooperative":
        if pipeline_config.reward_normalization.separate_norm_for_selfplay:
            logger.warning(
                "Forcing separate_norm_for_selfplay=False for microstep_cooperative. "
                "This advantage estimator performs turn-position-wise normalization internally."
            )
        # Force to False regardless of config
        use_separate_norm = False
    else:
        use_separate_norm = pipeline_config.reward_normalization.separate_norm_for_selfplay
    
    # Check if we should normalize players separately in self-play mode
    if use_separate_norm:
        rewards = reward_normalize_by_player(
            data=data,
            rewards=rewards, 
            rn_cfg=pipeline_config.reward_normalization,
            running_ctrl=running_ctrl,
            mask=mask,
            use_turn_level=use_turn_level_norm
        )
    else:
        rewards = score_normalize(
            rewards,
            rn_cfg=pipeline_config.reward_normalization,
            running_ctrl=running_ctrl,
            mask=mask,
        )

    # 2. clip
    if pipeline_config.reward_clip:
        reward_clip_frac = compute_clip_fraction(
            values=rewards,
            clip_max=pipeline_config.reward_clip,
            clip_min=-pipeline_config.reward_clip,
        )
        rewards = torch.clamp(
            rewards,
            min=-pipeline_config.reward_clip,
            max=pipeline_config.reward_clip,
        )
        metrics = {"critic/reward_clip_frac": reward_clip_frac}

    if pipeline_config.use_turn_scores:
        data.batch["token_level_rewards"] = rewards[:, 1:]
    else:
        data.batch["response_level_rewards"] = rewards
        data.batch["token_level_rewards"] = expand_to_token_level(data)

    # 3. compute token-level kl
    # TODO: (yhn) Here, kl is used as a per-token reward/penalty for rl.
    # We should consider using a separate (and differentiable) kl loss in the future,
    # as mentioned in the deepseek grpo paper: https://arxiv.org/abs/2402.03300
    token_level_rewards = data.batch["token_level_rewards"]
    if pipeline_config.add_token_level_kl:
        if kl_ctrl is not None and "ref_log_probs" in data.batch.keys():
            kld = compute_approx_kl(
                log_probs=data.batch["old_log_probs"],
                log_probs_base=data.batch["ref_log_probs"],
                action_mask=data.batch["response_mask"][:, 1:],
                kl_penalty=pipeline_config.kl_penalty,
            )
            beta = kl_ctrl.value
        else:
            kld = torch.zeros_like(data.batch["response_mask"][:, 1:], dtype=torch.float32)
            beta = 0
        token_level_rewards = token_level_rewards - beta * kld
        data.batch["token_level_rewards"] = token_level_rewards
        current_kl = masked_mean(kld, mask=data.batch["response_mask"][:, 1:], dim=-1)
        current_kl = torch.mean(current_kl, dim=0).item()
        kl_ctrl.update(current=current_kl, n_steps=data.batch.batch_size[0])
        metrics["critic/kl"] = current_kl
        metrics["critic/kl_coef"] = beta

    return data, metrics


@torch.no_grad()
def apply_kl_penalty(data: "DataProto", kl_ctrl: AdaptiveKLController, kl_penalty="kl"):
    """
    Apply KL divergence penalty to rewards to prevent policy from deviating too far from reference.
    
    This implements the KL-constrained RL objective:
        r'_t = r_t - beta * KL(pi || pi_ref)
    where beta is adaptively adjusted based on observed KL divergence.
    
    KL penalty types:
    - "kl": Standard KL divergence D_KL(pi || pi_ref)
    - "abs": Absolute difference |log pi - log pi_ref|
    - "mse": Mean squared error (log pi - log pi_ref)^2
    - "full": Full KL with both forward and reverse terms
    
    Args:
        data: DataProto containing old_log_probs and ref_log_probs
        kl_ctrl: Adaptive controller for KL coefficient beta
        kl_penalty: Type of KL penalty to apply
        
    Returns:
        data: Updated DataProto with KL-penalized token_level_rewards
        metrics: Dictionary with kl and kl_coef for logging
    """
    response_mask = data.batch["response_mask"][:, 1:]

    token_level_rewards = expand_to_token_level(data)
    if "token_level_rewards" in data.batch.keys():
        data.rename(old_keys="token_level_rewards", new_keys="token_level_scores")

    batch_size = data.batch.batch_size[0]

    if "ref_log_probs" in data.batch.keys():
        kld = compute_approx_kl(
            log_probs=data.batch["old_log_probs"],
            log_probs_base=data.batch["ref_log_probs"],
            action_mask=response_mask,
            kl_penalty=kl_penalty,
        )  # (batch_size, seq_len-1)
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_rewards - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    kl_ctrl.update(current=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coef": beta}

    return data, metrics


@torch.no_grad()
def compute_advantage(
    data: "DataProto",
    gamma,
    lambd,
    adv_estimator,
    advantage_clip=None,
    whiten_advantages=False,
    whiten_rewards=False,
    advantage_norm=None,
    response_mask=None,
):
    """
    Compute advantages for policy gradient training using various estimators.
    
    This is the main entry point for advantage computation in agentic RL.
    Supports multiple estimation methods:
    - REINFORCE: Monte Carlo returns (high variance, unbiased)
    - GAE: Generalized Advantage Estimation (bias-variance tradeoff)
    - GRPO: Group Relative Policy Optimization (REINFORCE + group normalization)
    
    Processing steps:
    1. Whiten rewards if enabled (zero mean, unit variance normalization)
    2. Compute advantages using selected estimator
    3. Normalize unique advantage values (for multi-turn consistency)
    4. Whiten advantages if enabled
    5. Clip advantages if specified
    
    Args:
        data: DataProto containing batch with token_level_rewards, values (for GAE), etc.
        gamma: Discount factor (0.0-1.0, typically 0.99 or 1.0)
        lambd: GAE lambda parameter (0.0-1.0, typically 0.95)
        adv_estimator: Advantage estimation method ("reinforce", "gae", or "grpo")
        advantage_clip: Max absolute advantage value for clipping (None = no clipping)
        whiten_advantages: If True, normalize advantages to zero mean and unit variance
        whiten_rewards: If True, normalize rewards before computing advantages
        advantage_norm: Mode for normalizing unique advantages ("mean" or "mean_std")
        response_mask: Mask for response tokens (None = use data.batch["response_mask"])
        
    Returns:
        data: Updated DataProto with advantages and returns in batch
        metrics: Dictionary of advantage-related metrics for logging
    """
    if response_mask is None:
        response_mask = data.batch["response_mask"][:, 1:]
    # import pdb; pdb.set_trace()
    token_level_rewards = data.batch["token_level_rewards"].float()
    if whiten_rewards:
        token_level_rewards = masked_whiten(values=token_level_rewards, mask=response_mask)
    # data.batch['token_level_rewards'] = token_level_rewards
    token_level_rewards = token_level_rewards * response_mask
    data.batch["token_level_rewards"] = token_level_rewards
    if adv_estimator == "gae":
        values = data.batch["values"].float()
        data.batch["values"] = values * response_mask
        advantages, returns = compute_gae_advantage_return(
            token_level_rewards=token_level_rewards, values=values, gamma=gamma, lambd=lambd
        )
    elif adv_estimator == "reinforce":
        advantages, returns = compute_reinforce_return(
            token_level_rewards=token_level_rewards, gamma=gamma, lambd=lambd
        )
    elif adv_estimator == "grpo":
        # NOTE: For grpo, remember to manually setting AgenticConfig.reward_normalization to mean_std
        advantages, returns = compute_reinforce_return(
            token_level_rewards=token_level_rewards, gamma=gamma, lambd=lambd
        )
    elif adv_estimator == "microstep_cooperative":
        # Multi-agent microstep for COOPERATIVE games (e.g., Hanabi): normalize per-player turns across games
        # print("\n[COMPUTE_ADVANTAGE DEBUG] Using microstep_cooperative estimator (COOPERATIVE)")
        turn_end_positions = data.batch.get("turn_end_positions", None)
        if turn_end_positions is None:
            # print("[COMPUTE_ADVANTAGE DEBUG] No turn_end_positions found, falling back to REINFORCE")
            advantages, returns = compute_reinforce_return(
                token_level_rewards=token_level_rewards, gamma=gamma, lambd=lambd
            )
        else:
            # Align turn_end_positions with token_level_rewards shape
            # token_level_rewards is already sliced to [:, 1:] in reward_postprocess_agentic
            # turn_end_positions needs the same slice
            if turn_end_positions.shape[1] != token_level_rewards.shape[1]:
                turn_end_positions = turn_end_positions[:, 1:]  # Align with response_mask dimensions
            
            # Validate shapes match
            assert turn_end_positions.shape == token_level_rewards.shape, \
                f"Shape mismatch: turn_end_positions {turn_end_positions.shape} vs token_level_rewards {token_level_rewards.shape}"
            
            advantages, returns = compute_cooperative_microstep_return(
                data=data,
                token_level_rewards=token_level_rewards,
                turn_end_positions=turn_end_positions,
                gamma=gamma,
                lambd=lambd,
            )
    else:
        raise NotImplementedError

    # Validate that advantages shape matches response_mask
    if advantages.shape != response_mask.shape:
        raise ValueError(
            f"Shape mismatch after advantage computation: "
            f"advantages {advantages.shape} vs response_mask {response_mask.shape}"
        )

    data.batch["raw_advantages"] = advantages
    # Normalize all unique advantage values
    if advantage_norm:
        advantages = normalize_unique_values_by_player(advantages, data, mode=advantage_norm)
    if whiten_advantages:
        # TODO: Should we consider response length during whitening?
        advantages = masked_whiten(values=advantages, mask=response_mask)
    advantages = advantages * response_mask

    if advantage_clip is not None:
        adv_clip_frac = compute_clip_fraction(values=advantages, clip_min=-advantage_clip, clip_max=advantage_clip)
        data.meta_info["metrics"] = {"critic/advantage_clip_frac": adv_clip_frac}
        advantages = torch.clamp(advantages, min=-advantage_clip, max=advantage_clip)

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class GenerateRequestType(enum.Enum):
    ADD = enum.auto()
    ABORT = enum.auto()
    STOP = enum.auto()
    ALIVE_CHECK = enum.auto()


def postprocess_generate(
    prompts: "DataProto",
    output: torch.Tensor,
    num_return_sequences,
    sequence_length,
    eos_token_id,
    pad_token_id,
    fill_eos_token=False,
) -> "DataProto":
    from roll.distributed.scheduler.protocol import DataProto

    if fill_eos_token:
        # yali: if last token of output is not pad_token_id, replace with eos_token_id
        #  TODO: Need to ablate the impact of this change
        last_token_index = output.size(1) - 1
        need_replace_mask = output[:, last_token_index] != pad_token_id
        output[need_replace_mask, last_token_index] = eos_token_id

    input_ids = prompts.batch["input_ids"]  # (bs, prompt_length)
    attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
    prompt_id = prompts.batch.get("prompt_id", None)

    # input_batch_size * num_return_sequences
    output_batch_size = output.size(0)
    input_batch_size = input_ids.size(0)
    prompt_length = input_ids.size(1)

    output = pad_to_length(output, sequence_length, pad_token_id)

    assert output.shape[1] == sequence_length, f"output shape {output.shape} != {sequence_length}"

    prompt = output[:, :prompt_length].clone()  # (bs, prompt_length)
    response = output[:, prompt_length:].clone()  # (bs, response_length)

    attention_mask = (
        attention_mask.unsqueeze(1).repeat(1, num_return_sequences, 1).view(output_batch_size, prompt_length)
    )
    response_mask = get_pad_mask(response_id=response, pad_token=pad_token_id, dtype=attention_mask.dtype)
    attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

    position_ids = prompts.batch["position_ids"]
    # if is_num_return_sequences_expand=True, num_return_sequences here equals 1
    if position_ids.dim() == 3:  # qwen2vl mrope, maybe can support in other ways
        position_ids = (
            position_ids.unsqueeze(1)
            .repeat(1, num_return_sequences, 1, 1)
            .view(output_batch_size, *position_ids.shape[-2:])
        )
        delta_position_id = torch.arange(1, (sequence_length - prompt_length) + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, 1, -1).expand(output_batch_size, 3, -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        # left padding for prompt and right padding for response, to be converted
        # to right padding which is consistent with output
        output_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

    assert attention_mask.any(dim=1).all(), f"has all 0 attention_mask, {attention_mask} {input_ids}"
    first_one = attention_mask.float().argmax(dim=1)
    new_response_mask = torch.zeros_like(attention_mask)  # response mask for cat input_ids
    for i in range(output_batch_size):
        shift = first_one[i].item()
        if shift > 0:
            output[i, :-shift] = output[i, shift:].clone()
        else:
            output[i, :] = output[i, :].clone()
        valid_length = attention_mask[i].sum().int().item()
        response_length = response_mask[i].sum().int().item()
        attention_mask[i][:valid_length] = 1
        attention_mask[i][valid_length:] = 0
        new_response_mask[i][valid_length - response_length : valid_length] = 1
        if position_ids.dim() == 3 and shift > 0:
            # shift as output to convert to right padding
            # NOTE: left shift without clear right might lead to unclean values
            # in right part, which especially is the case when using long prompt
            # length and short response length. This usually makes no effect if
            # mask is right, while it might make trouble to for multi-modal model
            # like Qwen2-vl, since extra image_token would be left which might
            # cause error: Image features and image tokens do not match
            output_position_ids[i, ..., :-shift] = output_position_ids[i, ..., shift:].clone()
            # only clean in VLM(qwen2-vl) to make no effect on LLM
            if prompt_length > response_length:
                output[i, -shift:] = pad_token_id

    prompt_mask = (attention_mask == 1) & (new_response_mask == 0)
    if position_ids.dim() == 3:
        position_ids = output_position_ids
    else:  # normal position_ids
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)
    batch = TensorDict(
        {
            "prompts": prompt,
            "responses": response,
            "input_ids": output,  # right pad
            "attention_mask": attention_mask,  # right pad
            "position_ids": position_ids,
            "prompt_mask": prompt_mask,
            "response_mask": new_response_mask,  # right pad, response tokens
        },
        batch_size=output_batch_size,
    )
    if prompt_id is not None:
        prompt_id = (
            prompt_id.squeeze().unsqueeze(1).repeat(1, num_return_sequences).view(output_batch_size, -1).squeeze(-1)
        )
        batch["prompt_id"] = prompt_id
    return DataProto(batch=batch)


def get_dist_info_from_comm_plan(comm_plan, rank_in_cluster, rank_in_worker):
    for src_rank, comm_plan_args in comm_plan.items():
        start_rank = 0
        for tgt_device in comm_plan_args["tgt_devices"]:
            start_rank += 1
            if tgt_device["rank"] == rank_in_cluster and tgt_device["device"]["rank"] == rank_in_worker:
                return start_rank, comm_plan_args
    return None, None


def separate_prompt_response(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, response_mask: torch.Tensor, pad_id: int
):
    prompt_mask = attention_mask.bool() & ~response_mask.bool()
    response_mask_valid = attention_mask.bool() & response_mask.bool()
    prompt_ids = torch.where(prompt_mask, input_ids, torch.full_like(input_ids, pad_id))
    response_ids = torch.where(response_mask_valid, input_ids, torch.full_like(input_ids, pad_id))
    return prompt_ids, response_ids
