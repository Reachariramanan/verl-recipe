# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DVPO Critic - DataParallel DVPO Critic with Distributional Value Estimation
"""

import logging
import os

import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import masked_mean
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import masked_mean as verl_masked_mean
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic import BasePPOCritic

from . import dvpo_core_algos

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


class DataParallelDVPOCritic(BasePPOCritic):
    """
    DataParallel DVPO Critic with Distributional Value Estimation
    Replaces scalar value prediction with full quantile distribution prediction
    """

    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"DVPO Critic use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.device_name = get_device_name()

        # DVPO-specific parameters
        self.n_quantiles = self.config.get("n_quantiles", 200)
        self.dvpo_alpha = self.config.get("dvpo_alpha", 0.1)
        self.dvpo_beta = self.config.get("dvpo_beta", 0.1)
        self.dvpo_loss_weights = self.config.get("dvpo_loss_weights", {
            "risk": 1.0,
            "cvar": 1.0,
            "gain": 1.0,
            "shift": 1.0,
            "shape": 1.0,
            "curv": 1.0,
            "consist": 1.0,
            "ablate_risk": False,
            "ablate_cvar": False,
            "ablate_gain": False,
            "ablate_shift": False,
            "ablate_shape": False,
            "ablate_curv": False,
            "ablate_consist": False,
        })

    def _forward_micro_batch(self, micro_batch):
        """
        Forward pass through distributional critic
        Returns quantile distributions instead of scalar values
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs
            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                # Forward pass through distributional critic
                output = self.critic_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                # Distributional critic returns quantiles: [total_nnz, n_quantiles]
                if hasattr(self.critic_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead - adapt to distributional
                    quantiles_rmpad = output[2].squeeze(0).unsqueeze(-1).expand(-1, self.n_quantiles)
                else:
                    # Our custom distributional critic
                    quantiles_rmpad = output  # [total_nnz, n_quantiles]

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    quantiles_rmpad = gather_outputs_and_unpad(
                        quantiles_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back - quantiles: [batch, seqlen, n_quantiles]
                quantiles = pad_input(quantiles_rmpad, indices=indices, batch=batch, seqlen=seqlen)
                # Select response part: [batch, response_length, n_quantiles]
                quantiles = quantiles[:, -response_length - 1 : -1]
            else:
                # Forward pass through distributional critic
                output = self.critic_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                if hasattr(self.critic_module, "v_head"):
                    # Adapt scalar critic to distributional
                    quantiles = output[2].unsqueeze(-1).expand(-1, -1, self.n_quantiles)
                else:
                    # Our custom distributional critic
                    quantiles = output  # [batch, seqlen, n_quantiles]

                quantiles = quantiles[:, -response_length - 1 : -1]  # [batch, response_length, n_quantiles]

            return quantiles

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dvpo critic", logger=logger)
    def compute_values(self, data: DataProto) -> torch.Tensor:
        """
        Compute distributional values (quantiles)
        Returns: [batch_size * response_length, n_quantiles]
        """
        self.critic_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = (
            ["responses", "input_ids", "response_mask", "attention_mask", "position_ids"]
            if "response_mask" in data.batch
            else ["responses", "input_ids", "attention_mask", "position_ids"]
        )
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        quantiles_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                quantiles = self._forward_micro_batch(model_inputs)
            quantiles_lst.append(quantiles)
        quantiles = torch.concat(quantiles_lst, dim=0)

        if use_dynamic_bsz:
            quantiles = restore_dynamic_batch(quantiles, batch_idx_list)

        # Flatten to [total_tokens, n_quantiles] for compatibility
        quantiles = quantiles.view(-1, self.n_quantiles)

        if "response_mask" in data.batch:
            response_mask = data.batch["response_mask"]
            response_mask = response_mask.to(quantiles.device)
            response_mask = response_mask.view(-1)  # Flatten
            # For distributional critic, we mask by setting to zero (could be improved)
            quantiles = quantiles * response_mask.unsqueeze(-1)

        return quantiles

    @GPUMemoryLogger(role="dvpo critic", logger=logger)
    def update_critic(self, data: DataProto):
        """
        Update DVPO critic using distributional loss
        """
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {
            "critic/vf_loss": 0.0,
            "critic/dvpo_loss": 0.0,
        }

        select_keys = ["input_ids", "responses", "response_mask", "attention_mask", "position_ids", "values", "returns"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the critic
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.critic_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    values = model_inputs["values"]  # Target scalar values
                    returns = model_inputs["returns"]  # Target scalar returns

                    # Get predicted quantiles
                    pred_quantiles = self._forward_micro_batch(model_inputs)  # [batch, seq_len, n_quantiles]

                    # Create target quantiles by expanding scalar targets
                    # For now, use the same value for all quantiles (can be improved with distributional targets)
                    batch_size, seq_len = pred_quantiles.shape[:2]
                    target_quantiles = returns.view(batch_size, seq_len, 1).expand(-1, -1, self.n_quantiles)

                    # Flatten for loss computation
                    pred_flat = pred_quantiles.view(-1, self.n_quantiles)
                    target_flat = target_quantiles.view(-1, self.n_quantiles)
                    response_mask_flat = response_mask.view(-1)

                    # Apply response mask
                    valid_mask = response_mask_flat > 0
                    if valid_mask.any():
                        pred_valid = pred_flat[valid_mask]
                        target_valid = target_flat[valid_mask]

                        # Compute DVPO loss
                        dvpo_loss = dvpo_core_algos.dvpo_critic_loss(
                            critic=self.critic_module,
                            states=pred_valid,  # This is a bit of a hack - should pass proper state representation
                            target_quantiles=target_valid,
                            weights=self.dvpo_loss_weights,
                            alpha=self.dvpo_alpha,
                            beta=self.dvpo_beta
                        )

                        if self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            loss_scale_factor = response_mask_flat.sum().item() / self.config.ppo_mini_batch_size
                            loss = dvpo_loss * loss_scale_factor
                        else:
                            loss_scale_factor = 1 / self.gradient_accumulation
                            loss = dvpo_loss * loss_scale_factor

                        loss.backward()

                        micro_batch_metrics.update({
                            "critic/dvpo_loss": dvpo_loss.detach().item(),
                            "critic/quantile_mean": pred_valid.mean().detach().item(),
                        })

                        metrics["critic/dvpo_loss"] += dvpo_loss.detach().item() * loss_scale_factor

                        # Add some quantile statistics
                        append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"critic/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        self.critic_optimizer.zero_grad()
        return metrics
