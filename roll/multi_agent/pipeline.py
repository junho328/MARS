"""
Multi-Agent Agentic Pipeline

Extends the AgenticPipeline to support multi-agent LoRA training
with separate adapters per agent.
"""

import json
import os
from typing import Any, Dict, List

import ray
import torch
from codetiming import Timer

from roll.agentic.rollout.rollout_scheduler import RolloutScheduler
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.multi_agent.config import MultiAgentConfig
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline, compute_data_metrics
from roll.pipeline.agentic.utils import dump_rollout_render
from roll.utils.functionals import (
    reward_postprocess_agentic,
    compute_advantage,
    reduce_metrics,
    masked_mean,
    RunningMoments,
    agg_loss,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger

logger = get_logger()


class MultiAgentAgenticPipeline(AgenticPipeline):
    """
    Agentic pipeline with multi-agent LoRA support.
    
    Key differences from AgenticPipeline:
    1. Creates MultiAgentActorWorker instead of ActorWorker
    2. Uses MultiAgentDeepSpeedStrategy that applies PEFT before DeepSpeed
    3. Injects multi-agent config BEFORE worker initialization
    4. Handles per-agent metrics aggregation
    """
    
    def __init__(
        self, 
        pipeline_config: AgenticConfig,
        multi_agent_config: MultiAgentConfig,
    ):
        from roll.distributed.executor.cluster import Cluster
        from roll.pipeline.base_pipeline import BasePipeline
        
        # Store multi-agent config
        self.multi_agent_config = multi_agent_config
        
        # Modify config if multi-agent is enabled
        if multi_agent_config.enabled:
            # Use our multi-agent worker
            pipeline_config.actor_train.worker_cls = (
                "roll.multi_agent.worker.MultiAgentActorWorker"
            )
            # Use our multi-agent DeepSpeed strategy
            pipeline_config.actor_train.strategy_args.strategy_name = (
                "multi_agent_deepspeed_train"
            )
            logger.info(
                f"Multi-agent training enabled with {multi_agent_config.num_agents} agents, "
                f"using multi_agent_deepspeed_train strategy"
            )
        
        # === BEGIN: Copy of AgenticPipeline.__init__ with modifications ===
        # Call BasePipeline init (skipping AgenticPipeline's)
        BasePipeline.__init__(self, pipeline_config)
        self.pipeline_config: AgenticConfig
        
        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)
        
        self.tokenizer = default_tokenizer_provider(
            model_args=self.pipeline_config.actor_train.model_args
        )
        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )
        
        # Create clusters
        self.actor_train = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )
        self.actor_infer = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        self.reference = Cluster(
            name=self.pipeline_config.reference.name,
            worker_cls=self.pipeline_config.reference.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.reference,
        )
        if self.pipeline_config.adv_estimator == "gae":
            self.critic = Cluster(
                name=self.pipeline_config.critic.name,
                worker_cls=self.pipeline_config.critic.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.critic,
            )
        
        self.train_rollout_scheduler = RolloutScheduler(
            config=self.pipeline_config,
            env_manager_config=self.pipeline_config.train_env_manager,
            resource_manager=self.resource_manager,
            infer_cluster=self.actor_infer,
            mode="train",
        )
        self.val_rollout_scheduler = RolloutScheduler(
            config=self.pipeline_config,
            env_manager_config=self.pipeline_config.val_env_manager,
            resource_manager=self.resource_manager,
            infer_cluster=self.actor_infer,
            mode="val",
        )
        
        # CRITICAL: Pass multi-agent config to workers BEFORE initialization
        if multi_agent_config.enabled:
            self._send_multi_agent_config_to_workers()
        
        # Now initialize workers
        refs = []
        refs.extend(self.actor_train.initialize(
            pipeline_config=self.pipeline_config, blocking=False
        ))
        if self.pipeline_config.adv_estimator == "gae":
            refs.extend(self.critic.initialize(
                pipeline_config=self.pipeline_config, blocking=False
            ))
        ray.get(refs)
        
        self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)
        self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True)
        
        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )
        
        if self.pipeline_config.adv_estimator == "gae":
            self.set_checkpoint_clusters(self.actor_train, self.critic)
        else:
            self.set_checkpoint_clusters(self.actor_train)
        
        self.running = {}
        # === END: Copy of AgenticPipeline.__init__ ===
        
        logger.info("MultiAgentAgenticPipeline initialized successfully")
    
    def _send_multi_agent_config_to_workers(self):
        """Send multi-agent config to workers BEFORE they initialize."""
        refs = []
        for worker in self.actor_train.workers:
            ref = worker.set_multi_agent_config_before_init.remote(self.multi_agent_config)
            refs.append(ref)
        ray.get(refs)
        logger.info(f"Sent multi-agent config to {len(refs)} workers before initialization")
    
    @torch.no_grad()
    def run(self):
        """
        Main training loop with multi-agent support.
        
        Extends the parent run() method with per-agent metrics tracking.
        """
        from ray.util.timer import _Timer
        
        tps_timer = _Timer(window_size=5)
        
        # Per-agent running moments for reward normalization
        if self.multi_agent_config.enabled:
            self.per_agent_running = {
                i: RunningMoments() for i in range(self.multi_agent_config.num_agents)
            }
        
        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            
            logger.info(f"Multi-agent pipeline step {global_step} start...")
            metrics = {}
            
            with tps_timer:
                # Standard pipeline operations
                if self.pipeline_config.adv_estimator == "gae":
                    self.critic.offload_states(blocking=True)
                self.actor_train.offload_states(blocking=True)
                
                model_update_metrics: Dict = self.model_update(global_step)
                metrics.update(model_update_metrics)
                
                batch: DataProto = DataProto()
                batch.meta_info = {"global_step": global_step}
                
                # Evaluation
                if global_step % self.pipeline_config.eval_steps == 0:
                    batch.meta_info["is_offload_states"] = False
                    eval_batch = self.val_rollout_scheduler.get_batch(
                        batch, self.pipeline_config.val_batch_size
                    )
                    eval_metrics = reduce_metrics(eval_batch.meta_info.get("metrics", {}))
                    
                    # Per-agent evaluation metrics
                    if self.multi_agent_config.enabled:
                        agent_eval_metrics = self._compute_per_agent_metrics(
                            eval_batch, prefix="val"
                        )
                        eval_metrics.update(agent_eval_metrics)
                    
                    eval_score = eval_batch.batch["scores"].sum(-1)
                    eval_metrics["score/mean"] = torch.mean(eval_score).detach().item()
                    eval_metrics["score/max"] = torch.max(eval_score).detach().item()
                    eval_metrics["score/min"] = torch.min(eval_score).detach().item()
                    metrics.update({f"val/{k}": v for k, v in eval_metrics.items()})
                    
                    del eval_batch
                
                # Rollout
                with Timer(name="rollout", logger=None) as rollout_timer:
                    batch.meta_info["is_offload_states"] = True
                    batch = self.train_rollout_scheduler.get_batch(
                        batch, self.pipeline_config.rollout_batch_size
                    )
                    batch.non_tensor_batch.pop("frames", None)
                
                metrics["time/rollout"] = rollout_timer.last
                metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                batch.meta_info["global_step"] = global_step
                
                # Reference log probs
                with Timer(name="cal_ref_log_probs", logger=None) as cal_timer:
                    ref_log_probs_refs = self.reference.compute_log_probs(batch, blocking=False)
                    ref_log_probs = DataProto.materialize_concat(data_refs=ref_log_probs_refs)
                    ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                    batch = batch.union(ref_log_probs)
                    avg_ref_log_prob = masked_mean(
                        batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:]
                    )
                    metrics.update(reduce_metrics(ref_log_probs.meta_info.pop("metrics", {})))
                    metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})
                metrics["time/ref_log_probs"] = cal_timer.last
                
                # Old log probs
                with Timer(name="cal_old_log_probs", logger=None) as cal_old_timer:
                    batch.meta_info["is_offload_states"] = False
                    old_log_probs_refs = self.actor_train.compute_log_probs(batch, blocking=False)
                    if self.pipeline_config.adv_estimator == "gae":
                        values_refs = self.critic.compute_values(batch, blocking=False)
                    old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
                    if self.pipeline_config.adv_estimator == "gae":
                        values = DataProto.materialize_concat(data_refs=values_refs)
                        batch = batch.union(values)
                        metrics.update(reduce_metrics(values.meta_info.pop("metrics", {})))
                    batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                    avg_old_log_prob = masked_mean(
                        batch.batch["old_log_probs"], batch.batch["response_mask"][:, 1:]
                    )
                    metrics.update({"critic/old_log_prob/mean": avg_old_log_prob.item()})
                    
                    agg_entropy = agg_loss(
                        loss_mat=old_log_probs.batch["entropy"],
                        loss_mask=batch.batch["response_mask"][:, 1:],
                        loss_agg_mode="token-mean",
                    )
                    metrics.update({"critic/entropy/mean": agg_entropy.item()})
                    metrics.update(reduce_metrics(old_log_probs.meta_info.pop("metrics", {})))
                metrics["time/old_log_probs"] = cal_old_timer.last
                
                # Advantage computation with per-agent normalization
                batch.batch["prompt_id"] = torch.arange(
                    batch.batch.batch_size[0], device=batch.batch.device
                )
                
                with Timer(name="adv", logger=None) as timer:
                    grouping = self.pipeline_config.reward_normalization.grouping
                    batch_grouped = {"default": batch}
                    if grouping != "batch":
                        batch_grouped = batch.group_by(keys=grouping)
                    
                    batch_list = []
                    for group_name, group_batch in batch_grouped.items():
                        if group_name not in self.running:
                            if self.pipeline_config.reward_normalization.separate_norm_for_selfplay:
                                self.running[group_name] = {
                                    "player_0": RunningMoments(),
                                    "player_1": RunningMoments()
                                }
                            else:
                                self.running[group_name] = RunningMoments()
                        
                        # Get rewards
                        scores = group_batch.batch["scores"].clone()
                        group_batch.batch["token_level_rewards"] = scores
                        penalty = group_batch.batch["penalty"]
                        acc_scores = scores.sum(dim=-1)
                        group_batch.batch["response_level_rewards"] = acc_scores + penalty
                        
                        # Reward postprocessing
                        group_batch, group_metrics = reward_postprocess_agentic(
                            data=group_batch,
                            pipeline_config=self.pipeline_config,
                            running_ctrl=self.running[group_name],
                            kl_ctrl=self.kl_ctrl,
                        )
                        metrics.update(group_metrics)
                        batch_list.append(group_batch)
                    
                    batch = DataProto.concat(batch_list)
                    batch.reorder(indices=torch.argsort(batch.batch["prompt_id"]))
                    batch.pop("prompt_id")
                    
                    # Compute advantages
                    batch = compute_advantage(
                        data=batch,
                        gamma=self.pipeline_config.gamma,
                        lambd=self.pipeline_config.lambd,
                        adv_estimator=self.pipeline_config.adv_estimator,
                        advantage_clip=self.pipeline_config.advantage_clip,
                        whiten_advantages=self.pipeline_config.whiten_advantages,
                        whiten_rewards=self.pipeline_config.whiten_rewards,
                        advantage_norm=self.pipeline_config.advantage_norm,
                    )
                
                metrics["time/adv"] = timer.last
                
                # Critic training
                if self.pipeline_config.adv_estimator == "gae":
                    critic_train_refs = self.critic.train_step(batch, blocking=False)
                
                # Actor training (multi-agent aware)
                if self.pipeline_config.critic_warmup <= global_step:
                    actor_train_refs = self.actor_train.train_step(batch, blocking=False)
                    actor_train_metrics = DataProto.materialize_concat(data_refs=actor_train_refs)
                    
                    # Extract per-agent metrics
                    actor_metrics = actor_train_metrics.meta_info.pop("metrics", {})
                    metrics.update(reduce_metrics(actor_metrics))
                
                if self.pipeline_config.adv_estimator == "gae":
                    critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_refs)
                    metrics.update(reduce_metrics(critic_train_metrics.meta_info.pop("metrics", {})))
                
                tps_timer.push_units_processed(
                    n=torch.sum(batch.batch["attention_mask"]).detach().item()
                )
            
            # Data metrics
            data_metrics = compute_data_metrics(batch=batch)
            metrics.update(data_metrics)
            
            # Per-agent data metrics
            if self.multi_agent_config.enabled:
                agent_data_metrics = self._compute_per_agent_metrics(batch, prefix="train")
                metrics.update(agent_data_metrics)
            
            metrics["system/tps"] = tps_timer.mean_throughput
            metrics["system/samples"] = (global_step + 1) * batch.batch.shape[0]
            
            # Checkpoint and logging
            self.state.step = global_step
            self.state.log_history.append(metrics)
            self.do_checkpoint(global_step=global_step)
            self.tracker.log(values=metrics, step=global_step)
            
            if global_step % self.pipeline_config.logging_steps == 0:
                self._log_samples(batch, global_step)
            
            logger.info(f"Multi-agent pipeline step {global_step} finished")
            global_step += 1
        
        logger.info("Multi-agent pipeline complete!")
    
    def _compute_per_agent_metrics(
        self, 
        batch: DataProto, 
        prefix: str = "train"
    ) -> Dict[str, float]:
        """Compute per-agent metrics from a batch."""
        metrics = {}
        
        if "group_ids" not in batch.non_tensor_batch:
            return metrics
        
        group_ids = batch.non_tensor_batch["group_ids"]
        scores = batch.batch["scores"].sum(-1)
        
        for agent_id in range(self.multi_agent_config.num_agents):
            agent_mask = torch.tensor([
                self.multi_agent_config.agent_id_from_group_id(str(gid)) == agent_id
                for gid in group_ids
            ], device=scores.device)
            
            if agent_mask.sum() == 0:
                continue
            
            agent_scores = scores[agent_mask]
            metrics[f"{prefix}/agent_{agent_id}/score/mean"] = agent_scores.mean().item()
            metrics[f"{prefix}/agent_{agent_id}/score/max"] = agent_scores.max().item()
            metrics[f"{prefix}/agent_{agent_id}/score/min"] = agent_scores.min().item()
            metrics[f"{prefix}/agent_{agent_id}/num_samples"] = agent_mask.sum().item()
        
        return metrics
    
    def _log_samples(self, batch: DataProto, global_step: int):
        """Log sample outputs."""
        prompt_mask = batch.batch["prompt_mask"]
        non_prompt_mask = batch.batch["non_prompt_mask"]
        input_ids = batch.batch["input_ids"]
        
        prompt_ids = torch.where(
            prompt_mask.bool(), input_ids, 
            torch.full_like(input_ids, self.tokenizer.pad_token_id)
        )
        response_ids = torch.where(
            non_prompt_mask.bool(), input_ids,
            torch.full_like(input_ids, self.tokenizer.pad_token_id)
        )
        
        prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        episode_scores = batch.non_tensor_batch["episode_scores"].tolist()
        
        generate_res = []
        for prompt, response, score in zip(prompts[:10], responses[:10], episode_scores[:10]):
            generate_res.append({
                "prompt": prompt[:200],
                "response": response[:200],
                "episode_score": score,
            })
        
        logger.info(f"Training samples at step {global_step}:")
        logger.info(json.dumps(generate_res[:5], ensure_ascii=False, indent=2))


def create_multi_agent_pipeline(
    pipeline_config: AgenticConfig,
    multi_agent_config: MultiAgentConfig,
) -> MultiAgentAgenticPipeline:
    """
    Factory function to create a multi-agent pipeline.
    
    Args:
        pipeline_config: The agentic pipeline configuration
        multi_agent_config: The multi-agent configuration
        
    Returns:
        A configured MultiAgentAgenticPipeline
    """
    return MultiAgentAgenticPipeline(pipeline_config, multi_agent_config)

