import os
import threading
import time
from typing import Union, Optional, Dict, List

import numpy as np
import ray
import torch
from codetiming import Timer
from tqdm import tqdm

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import register, Dispatch
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_actor_model_provider, default_value_model_provider, \
    default_reward_model_provider
from roll.utils.context_managers import state_offload_manger
from roll.utils.functionals import (
    append_to_dict,
    masked_mean,
    compute_approx_kl,
    postprocess_generate,
    GenerateRequestType,
    agg_loss,
)
from roll.utils.offload_states import OffloadStateType


class ActorWorker(Worker):
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None
        self.response_call_back_fns = {}
        self.response_callback_refs = []
        self.server_metrics = {}
        self.thread_server = None
        self.offload_manager = None
        # Multi-adapter support
        self.adapter_manager = None
        self.multi_adapter_mode = False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        self.strategy.initialize(model_provider=default_actor_model_provider)
        self.tokenizer = self.strategy.tokenizer
        
        # Check if multi-adapter mode is enabled and store adapter_manager
        if hasattr(self.strategy, 'adapter_manager') and self.strategy.adapter_manager is not None:
            self.adapter_manager = self.strategy.adapter_manager
            self.multi_adapter_mode = True
            self.logger.info(f"Multi-adapter mode enabled with {self.adapter_manager.num_agents} agents")
        
        if self.pipeline_config.resume_from_checkpoint:
            load_dir = self.pipeline_config.resume_from_checkpoint
            self.strategy.load_checkpoint(load_dir=load_dir, tag="checkpoint")
        self.logger.info(f"{self.worker_name} initialized")

        self.strategy.offload_states()

        # Cuda must have been initialized when calling torch.cuda.reset_max_memory_allocated
        # with arguments (inside state_offload_manager). We explicitly init cuda here because
        # current process is used as engine client when using vllm v1 engine, and
        # there is no chance to init cuda context.
        torch.cuda.init()

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def train_step(self, data: DataProto):
        """
        return DataProto(meta_info={'metrics': metrics})
        
        For multi-adapter mode, this method groups data by agent_id and trains
        each adapter separately to ensure gradient isolation.
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        self.logger.info(f"{self.worker_name} generate global step {global_step}")

        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/train_step",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params, OffloadStateType.other_params]},
        ):
            data = data.to("cuda")
            data = self.strategy.get_data_input(data)
            per_device_train_batch_size = self.worker_config.training_args.per_device_train_batch_size
            backward_batch_size = (
                per_device_train_batch_size * self.worker_config.training_args.gradient_accumulation_steps
            )

            if self.multi_adapter_mode:
                # Multi-adapter training: group by agent_id and train each adapter separately
                metrics = self._train_step_multi_adapter(
                    data=data,
                    backward_batch_size=backward_batch_size,
                    global_step=global_step,
                    metrics=metrics,
                )
            else:
                # Standard single-adapter training
                dataloader = data.make_iterator(
                    mini_batch_size=backward_batch_size,
                    epochs=self.pipeline_config.ppo_epochs,
                    dataloader_kwargs={"shuffle": True},
                )

                for batch_idx, batch_data in tqdm(
                    enumerate(dataloader),
                    desc=f"{self.worker_name} train global step {global_step}",
                    total=data.batch.batch_size[0] * self.pipeline_config.ppo_epochs // backward_batch_size,
                ):
                    pg_metrics = self.strategy.train_step(batch=batch_data, loss_func=self.loss_func)
                    append_to_dict(metrics, pg_metrics)

            metrics["actor/lr"] = self.strategy.scheduler.get_last_lr()[0]
            data.to("cpu")

        output = DataProto(meta_info={"metrics": metrics})
        return output
    
    def _train_step_multi_adapter(self, data: DataProto, backward_batch_size: int, global_step: int, metrics: Dict):
        """
        Train step for multi-adapter mode.
        
        Groups data by agent_id and trains each adapter separately.
        This ensures gradient isolation - each adapter only receives
        gradients from its own agent's data.
        """
        # Get agent_ids from data
        if "agent_ids" not in data.non_tensor_batch:
            self.logger.warning("agent_ids not found in data, falling back to single adapter training")
            # Fallback: train with first adapter
            self.adapter_manager.activate_adapter(0, enable_gradient=True)
            dataloader = data.make_iterator(
                mini_batch_size=backward_batch_size,
                epochs=self.pipeline_config.ppo_epochs,
                dataloader_kwargs={"shuffle": True},
            )
            for batch_idx, batch_data in enumerate(dataloader):
                pg_metrics = self.strategy.train_step(batch=batch_data, loss_func=self.loss_func)
                append_to_dict(metrics, pg_metrics)
            return metrics
        
        agent_ids = data.non_tensor_batch["agent_ids"]
        unique_agents = set(agent_ids.flatten().tolist())
        
        self.logger.info(f"Multi-adapter training with agents: {unique_agents}")
        
        for agent_id in sorted(unique_agents):
            # Filter data for this agent
            agent_mask = (agent_ids == agent_id).flatten()
            if not agent_mask.any():
                continue
            
            # Create agent-specific data subset
            agent_data = data.select(indices=torch.where(agent_mask)[0])
            
            if agent_data.batch.batch_size[0] == 0:
                continue
            
            # Activate this agent's adapter (enables gradients only for this adapter)
            self.adapter_manager.activate_adapter(int(agent_id), enable_gradient=True)
            self.logger.debug(f"Training adapter for agent {agent_id} with {agent_data.batch.batch_size[0]} samples")
            
            # Create dataloader for this agent's data
            dataloader = agent_data.make_iterator(
                mini_batch_size=min(backward_batch_size, agent_data.batch.batch_size[0]),
                epochs=self.pipeline_config.ppo_epochs,
                dataloader_kwargs={"shuffle": True},
            )
            
            for batch_idx, batch_data in tqdm(
                enumerate(dataloader),
                desc=f"{self.worker_name} train agent {agent_id} step {global_step}",
                total=max(1, agent_data.batch.batch_size[0] * self.pipeline_config.ppo_epochs // backward_batch_size),
            ):
                pg_metrics = self.strategy.train_step(batch=batch_data, loss_func=self.loss_func)
                # Prefix metrics with agent_id for tracking
                agent_metrics = {f"agent_{agent_id}/{k}": v for k, v in pg_metrics.items()}
                append_to_dict(metrics, agent_metrics)
        
        return metrics

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def generate(self, data: DataProto):
        """
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'old_log_probs': log_probs,
            },
            batch_size=batch_size)
        return DataProto(batch=batch)
        """
        if "generation_config" not in data.meta_info:
            generation_config = self.worker_config.generating_args.to_dict()
        else:
            generation_config = data.meta_info["generation_config"]

        generation_config["eos_token_id"] = [
            self.tokenizer.eos_token_id
        ] + self.tokenizer.additional_special_tokens_ids
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id

        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        self.logger.info(f"{self.worker_name} generate global step {global_step}")

        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/generate",
            is_offload_states=is_offload_states,
        ):
            data = data.to("cuda")
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size

            output = self.strategy.generate(batch=data, generation_config=generation_config)
            output = postprocess_generate(
                prompts=data,
                output=output,
                num_return_sequences=generation_config["num_return_sequences"],
                sequence_length=self.pipeline_config.sequence_length,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            data.to("cpu")
            output = output.to("cpu")

        output.meta_info = {"metrics": metrics}
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL_ONE)
    @torch.no_grad()
    def start_server(self, data: DataProto):
        """
        解决dp generate的长尾问题，async+ load balance
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)

        self.logger.info(f"{self.worker_name} generate server global step {global_step}")
        self.response_call_back_fns = {}

        self.response_callback_refs = []
        self.server_metrics = {}
        self.offload_manager = state_offload_manger(
            strategy=self.strategy,
            metrics=self.server_metrics,
            metric_infix=f"{self.cluster_name}/generate",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params]},
        )
        self.offload_manager.__enter__()
        self.thread_server = threading.Thread(
            target=self.strategy.start_server, kwargs=dict(data=data, request_complete_callback=self.request_complete)
        )
        self.thread_server.start()
        while not self.strategy.running:
            time.sleep(0.1)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL_ONE)
    def stop_server(self, data: DataProto = None):
        if not hasattr(self, "thread_server"):
            raise ValueError("server is not initialized")

        self.strategy.add_request(command=GenerateRequestType.STOP, data=data)
        self.thread_server.join()
        self.thread_server = None
        self.response_call_back_fns.clear()
        self.offload_manager.__exit__(None, None, None)
        ray.get(self.response_callback_refs)
        self.response_callback_refs.clear()

        return DataProto(meta_info={"metrics": self.server_metrics})

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def compute_log_probs(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'log_probs': output})
        
        For multi-adapter mode, computes log probs for each agent using
        the corresponding adapter.
        """
        data = self.strategy.get_data_input(data)
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_log_probs",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            data = data.to("cuda")
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
            
            if self.multi_adapter_mode and "agent_ids" in data.non_tensor_batch:
                # Multi-adapter mode: compute log probs per agent
                results = self._compute_log_probs_multi_adapter(data)
            else:
                # Standard single-adapter mode
                with torch.no_grad():
                    results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                        batch=data, forward_func=self.forward_func_log_probs
                    )
            
            if results is None:
                return DataProto(batch=None, meta_info={"metrics": metrics})
            output = DataProto.from_dict(tensors={"log_probs": results["log_probs"], "entropy": results["entropy"]})
            output = output.to("cpu")
            data.to("cpu")
        output.meta_info = {"metrics": metrics}
        return output
    
    def _compute_log_probs_multi_adapter(self, data: DataProto) -> Dict[str, torch.Tensor]:
        """
        Compute log probs for multi-adapter mode.
        
        Processes each agent's data with the corresponding adapter to ensure
        log probs are computed with the correct adapter weights.
        """
        agent_ids = data.non_tensor_batch["agent_ids"]
        batch_size = data.batch.batch_size[0]
        seq_len = data.batch["input_ids"].shape[1]
        
        # Initialize output tensors
        all_log_probs = torch.zeros(batch_size, seq_len - 1, device=data.batch["input_ids"].device)
        all_entropy = torch.zeros(batch_size, seq_len - 1, device=data.batch["input_ids"].device)
        
        unique_agents = set(agent_ids.flatten().tolist())
        
        for agent_id in sorted(unique_agents):
            # Filter data for this agent
            agent_mask = (agent_ids == agent_id).flatten()
            agent_indices = torch.where(torch.tensor(agent_mask))[0]
            
            if len(agent_indices) == 0:
                continue
            
            # Create agent-specific data subset
            agent_data = data.select(indices=agent_indices)
            
            # Activate this agent's adapter (no gradient needed for inference)
            self.adapter_manager.activate_adapter(int(agent_id), enable_gradient=False)
            
            with torch.no_grad():
                results = self.strategy.forward_step(
                    batch=agent_data, forward_func=self.forward_func_log_probs
                )
            
            if results is not None:
                # Place results back in the correct positions
                for i, idx in enumerate(agent_indices):
                    all_log_probs[idx] = results["log_probs"][i]
                    all_entropy[idx] = results["entropy"][i]
        
        return {"log_probs": all_log_probs, "entropy": all_entropy}

    def forward_func_log_probs(self, data: DataProto, output_tensor: torch.Tensor):
        """
        forward func 接口定义:
            data: DataProto, 由forward_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
        )
        entropy = self.strategy.op_compute_entropy(logits=output_tensor, attention_mask=data.batch["response_mask"])
        return log_probs, {"log_probs": log_probs.clone().detach(), "entropy": entropy.clone().detach()}

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """

        response_mask = data.batch["response_mask"][:, 1:].long()
        ref_log_probs = data.batch["ref_log_probs"]
        old_log_probs = data.batch["old_log_probs"]
        advantages = data.batch["advantages"]

        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
        )

        ratio = (log_probs - old_log_probs).exp()

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.pipeline_config.pg_clip, 1 + self.pipeline_config.pg_clip) * advantages
        pg_loss = -torch.min(surr1, surr2)
        if self.pipeline_config.dual_clip_loss:
            dual_clip_loss = -torch.max(-pg_loss, (1 + self.pipeline_config.pg_clip * 2) * advantages)
            pg_loss = torch.where(advantages < 0, dual_clip_loss, pg_loss)

        pg_loss = agg_loss(loss_mat=pg_loss, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode)

        kl_loss = compute_approx_kl(log_probs=log_probs, log_probs_base=ref_log_probs, action_mask=response_mask,
                                    kl_penalty="k3")
        kl_loss = agg_loss(loss_mat=kl_loss, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode)

        approxkl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="mse"
        )
        policykl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="kl"
        )

        clipped_low = (ratio < 1 - self.pipeline_config.pg_clip).float()
        clipped_high = (ratio > 1 + self.pipeline_config.pg_clip_high).float()
        clipped = (clipped_low + clipped_high).float()

        entropy = self.strategy.op_compute_entropy(logits=output_tensor, attention_mask=data.batch["response_mask"])
        entropy_loss = agg_loss(
            loss_mat=entropy,
            loss_mask=response_mask,
            loss_agg_mode=self.pipeline_config.loss_agg_mode,
        )

        if self.pipeline_config.use_kl_loss:
            total_loss = pg_loss + kl_loss * self.pipeline_config.kl_loss_coef
        else:
            total_loss = pg_loss
        if self.pipeline_config.entropy_loss_coef > 0:
            total_loss = total_loss - entropy_loss * self.pipeline_config.entropy_loss_coef

        pg_metrics = {
            "actor/ppo_ratio_high_clipfrac": clipped_high.mean().detach().item(),
            "actor/ppo_ratio_low_clipfrac": clipped_low.mean().detach().item(),
            "actor/ppo_ratio_clipfrac": clipped.mean().detach().item(),
            "actor/ratio_mean": masked_mean(ratio, response_mask, dim=-1).mean().detach().item(),
            "actor/ratio_max": torch.max(ratio * response_mask).detach().item(),
            "actor/ratio_min": torch.min(ratio * response_mask + (1 - response_mask) * 1e10).detach().item(),
            "actor/clipfrac": agg_loss(loss_mat=torch.lt(surr2, surr1).float(), loss_mask=response_mask,
                                       loss_agg_mode=self.pipeline_config.loss_agg_mode).detach().item(),
            "actor/pg_loss": pg_loss.detach().item(),
            "actor/kl_loss": kl_loss.detach().item(),
            "actor/total_loss": total_loss.detach().item(),
            "actor/approxkl": agg_loss(loss_mat=approxkl, loss_mask=response_mask,
                                       loss_agg_mode=self.pipeline_config.loss_agg_mode).detach().item(),
            "actor/policykl": agg_loss(loss_mat=policykl, loss_mask=response_mask,
                                       loss_agg_mode=self.pipeline_config.loss_agg_mode).detach().item(),
        }

        return total_loss, pg_metrics

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def do_checkpoint(self, global_step):
        with Timer("do_checkpoint") as total_timer:
            ckpt_id = f"checkpoint-{global_step}"

            # actor train是直接存在save dir目录下的，其他role是存在save_dir/cluster_name下的
            save_dir = os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id)
            self.logger.info(f"save checkpoint-{global_step} to {save_dir}")

            exec_metrics: Dict = self.strategy.save_checkpoint(save_dir, global_step, ckpt_id)

        metrics = {
            f"time/{self.cluster_name}/do_checkpoint/total": total_timer.last,
        }
        metric_prefix = f"time/{self.cluster_name}/do_checkpoint"
        metrics.update({f"{metric_prefix}/{k}": v for k, v in exec_metrics.items()})
        output = DataProto(meta_info={"metrics": metrics})
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    def add_request(self, command, data: DataProto):
        """
        data req meta_info里需要包含:
            request_id: str
            response_callback_fn: callable
        generation_config, 按request设置
        """
        if command == GenerateRequestType.ALIVE_CHECK:
            if self.thread_server is not None:
                if not self.thread_server.is_alive():
                    raise Exception("thread server has stopped unexpectedly. check stderr for more info.")
            output = DataProto(meta_info={"request_counts": len(self.response_call_back_fns)})
            return output
        elif command == GenerateRequestType.ADD:
            assert "response_callback_fn" in data.meta_info, "response_callback_fn is not in data.meta_info"
            is_num_return_sequences_expand = data.meta_info.get("is_num_return_sequences_expand", False)
            if "generation_config" not in data.meta_info:
                generation_config = self.worker_config.generating_args.to_dict()
                if is_num_return_sequences_expand:
                    self.worker_config.generating_args.num_return_sequences = 1
                    generation_config["num_return_sequences"] = 1
                    self.logger.info(f"is_num_return_sequences_expand is True, set num_return_sequences to 1.")
            else:
                generation_config = data.meta_info["generation_config"]
            generation_config["eos_token_id"] = [
                self.tokenizer.eos_token_id
            ] + self.tokenizer.additional_special_tokens_ids
            generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            data.meta_info["generation_config"] = generation_config
            self.response_call_back_fns[data.meta_info["request_id"]] = data.meta_info.pop("response_callback_fn")
        self.strategy.add_request(command=command, data=data)
        return DataProto(meta_info={"request_counts": len(self.response_call_back_fns)})

    def request_complete(self, data: DataProto):
        data.meta_info["eos_token_id"] = self.tokenizer.eos_token_id
        data.meta_info["pad_token_id"] = self.tokenizer.pad_token_id
        response_call_back_fn = self.response_call_back_fns.pop(data.meta_info["request_id"])
        self.response_callback_refs.append(response_call_back_fn(data))


class CriticWorker(Worker):

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        self.strategy.initialize(model_provider=default_value_model_provider)
        self.tokenizer = self.strategy.tokenizer

        if self.pipeline_config.resume_from_checkpoint:
            load_dir = os.path.join(self.pipeline_config.resume_from_checkpoint, self.cluster_name)
            self.strategy.load_checkpoint(load_dir=load_dir, tag="checkpoint")

        self.logger.info(f"{self.worker_name} initialized")

        self.strategy.offload_states()

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def compute_values(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'values': values})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_values",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            data = data.to("cuda")
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
            with torch.no_grad():
                results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                    batch=data, forward_func=self.forward_func_values
                )

            output = DataProto.from_dict(tensors={"values": results["values"]})
            data.to("cpu")
            output = output.to("cpu")

        output.meta_info = {"metrics": metrics}
        return output

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def train_step(self, data: DataProto):
        """
        return DataProto(meta_info={'metrics': metrics})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/train_step",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params, OffloadStateType.other_params]},
        ):
            data = data.to("cuda")
            per_device_train_batch_size = self.worker_config.training_args.per_device_train_batch_size
            backward_batch_size = (
                per_device_train_batch_size * self.worker_config.training_args.gradient_accumulation_steps
            )

            dataloader = data.make_iterator(
                mini_batch_size=backward_batch_size, epochs=1, dataloader_kwargs={"shuffle": True}
            )

            for batch_idx, data in tqdm(
                enumerate(dataloader),
                desc=f"{self.worker_name} train global step {global_step}",
                total=data.batch.batch_size[0] * self.pipeline_config.ppo_epochs // backward_batch_size,
            ):
                vf_metrics = self.strategy.train_step(batch=data, loss_func=self.loss_func)
                append_to_dict(metrics, vf_metrics)

            data.to("cpu")
            metrics["critic/lr"] = self.strategy.scheduler.get_last_lr()[0]

        output = DataProto(meta_info={"metrics": metrics}).to("cpu")

        return output

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        response_mask = data.batch["response_mask"][:, 1:]
        old_values = data.batch["values"]
        returns = data.batch["returns"]

        values, _ = self.forward_func_values(data=data, output_tensor=output_tensor)

        if self.pipeline_config.value_clip is not None:
            values_clipped = torch.clip(
                values,
                old_values - self.pipeline_config.value_clip,
                old_values + self.pipeline_config.value_clip,
            )
            surr1 = (values - returns) ** 2
            surr2 = (values_clipped - returns) ** 2
            vf_clipfrac = masked_mean(torch.gt(surr2, surr1).float(), response_mask, dim=-1).mean()
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2
            vf_clipfrac = masked_mean(loss, response_mask, dim=-1).mean()

        vf_loss = 0.5 * masked_mean(loss, response_mask, dim=-1).mean()

        vf_metrics = {
            "critic/loss": vf_loss.detach().item(),
            "critic/value": (masked_mean(old_values, response_mask, dim=-1)).mean().detach().item(),
            "critic/vpred": (masked_mean(values, response_mask, dim=-1)).mean().detach().item(),
            "critic/clipfrac": vf_clipfrac.detach().item(),
            "critic/error": masked_mean((values - returns) ** 2, response_mask, dim=-1).mean().detach().item(),
        }

        return vf_loss, vf_metrics

    def forward_func_values(self, data: DataProto, output_tensor: torch.Tensor):
        values = output_tensor[:, :-1]
        values = values.squeeze(dim=-1)
        return values, {"values": values.clone().detach()}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def do_checkpoint(self, global_step):
        with Timer("do_checkpoint") as total_timer:
            ckpt_id = f"checkpoint-{global_step}"
            save_dir = os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id, self.cluster_name)
            self.logger.info(f"save checkpoint-{global_step} to {save_dir}")
            exec_metrics: Dict = self.strategy.save_checkpoint(save_dir, global_step, ckpt_id)

        metrics = {
            f"time/{self.cluster_name}/do_checkpoint/total": total_timer.last,
        }
        metric_prefix = f"time/{self.cluster_name}/do_checkpoint"
        metrics.update({f"{metric_prefix}/{k}": v for k, v in exec_metrics.items()})
        output = DataProto(meta_info={"metrics": metrics})
        return output


class RewardWorker(Worker):
    """
    Reward Model 使用 AutoModelForSequenceClassification 协议
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        self.strategy.initialize(model_provider=default_reward_model_provider)
        self.tokenizer = self.strategy.tokenizer

        self.logger.info(f"{self.worker_name} initialized")
        self.strategy.offload_states()

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'rewards': rewards})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_rewards",
            is_offload_states=is_offload_states,
        ):
            data = data.to("cuda")

            # TODO: _switch_chat_template, 异构reward model

            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
            with torch.no_grad():
                results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                    batch=data, forward_func=self.forward_func_values
                )
            token_level_rewards = results["values"]  # (bsz, input_ids.shape[1]-1)
            input_ids = data.batch["input_ids"][:, 1:]
            seq_lengths = torch.eq(input_ids, self.tokenizer.pad_token_id).int().argmax(-1) - 1
            seq_lengths = (seq_lengths % input_ids.shape[-1]).to(token_level_rewards.device)
            response_level_rewards = token_level_rewards[
                torch.arange(seq_lengths.shape[0], device=token_level_rewards.device), seq_lengths
            ]

            output = DataProto.from_dict(
                tensors={"token_level_rewards": token_level_rewards, "response_level_rewards": response_level_rewards}
            )

            data.to("cpu")
            output = output.to("cpu")

        output.meta_info = {"metrics": metrics}
        return output

    def forward_func_values(self, data: DataProto, output_tensor: torch.Tensor):
        values = output_tensor[:, 1:]
        values = values.squeeze(dim=-1)
        return values, {"values": values.clone().detach()}
