#
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
#
# This file is a part of the vllm-kunlun project.
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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE Triton kernels."""
import functools
# torch.compile needs typing.List. It will fail torch.library.infer_schema
# otherwise
from typing import List  # noqa: UP035
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F

import vllm.envs as envs
# yapf: disable
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG, FusedMoEQuantConfig, _get_config_dtype_str)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_moe_wna16_block_config,
    try_get_optimal_moe_config,
    fused_grouped_topk,
    inplace_fused_experts_fake,
    outplace_fused_experts_fake,
    _get_config_quant_dtype)
from vllm.model_executor.layers.fused_moe.cutlass_moe import (
    _valid_cutlass_block_scaled_grouped_gemm,
    run_cutlass_block_scaled_fused_experts)
# yapf: enable
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import (
    _valid_deep_gemm)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache, activation_without_mul, moe_kernel_quantize_input)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils import direct_register_custom_op, is_torch_equal_or_newer

from vllm_kunlun.ops.fused_moe.moe_align_block_size import (
    moe_align_block_size)
import kunlun_ops
import xspeedgate_ops


def fused_moe_kernel_gptq_awq_torch(
    # Pointers to matrices (实际是 PyTorch 张量)
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    # Config parameters
    block_k_diviable,
    group_size,
    # Meta-parameters
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    GROUP_SIZE_M,
    MUL_ROUTED_WEIGHT,
    top_k,
    compute_type,
    has_zp,
    use_int4_w4a16,
    use_int8_w8a16,
):
    """
    Native PyTorch implementation for reference.
    Mimics the logic of fused_moe_kernel_gptq_awq_triton_impl.

    This function signature is aligned with the Triton kernel implementation
    (fused_moe_kernel_gptq_awq_triton_impl) for seamless compatibility.

    Args:
        a_ptr: Input tensor [M, K]
        b_ptr: Quantized weight tensor [E, N, K_packed]
        c_ptr: Output tensor [M, top_k, N] (modified in-place)
        b_scale_ptr: Scale tensor [E, N, K//group_size]
        b_zp_ptr: Zero point tensor [E, N, K//group_size] or None
        topk_weights_ptr: Router weights [M, top_k] or None
        sorted_token_ids_ptr: Sorted token IDs [EM]
        expert_ids_ptr: Expert IDs per block [num_m_blocks]
        num_tokens_post_padded_ptr: Scalar tensor with padded token count
        N, K, EM, num_valid_tokens: Matrix dimensions
        stride_*: Stride parameters for tensor indexing
        block_k_diviable: Whether K is divisible by BLOCK_SIZE_K
        group_size: Quantization group size
        BLOCK_SIZE_M/N/K: Block sizes for computation
        GROUP_SIZE_M: Tiling parameter
        MUL_ROUTED_WEIGHT: Whether to apply router weights
        top_k: Top-k experts per token
        compute_type: Computation dtype (torch.float16/float32/bfloat16)
        has_zp: Whether zero points are provided
        use_int4_w4a16: Use int4 quantization
        use_int8_w8a16: Use int8 quantization

    Returns:
        None (modifies c_ptr in-place)
    """
    num_tokens_post_padded = num_tokens_post_padded_ptr.item()

    # Get device and dtype from input
    device = a_ptr.device
    dtype = a_ptr.dtype
    c_ptr.zero_()
    E = b_ptr.shape[0]
    num_m_blocks = expert_ids_ptr.shape[0]

    invalid_expert_mask = (expert_ids_ptr < -1) | (expert_ids_ptr >= E)
    if invalid_expert_mask.any():
        invalid_indices = expert_ids_ptr[invalid_expert_mask].tolist()
        raise ValueError(
            f"[fused_moe_kernel_gptq_awq_torch] Found {len(invalid_indices)} "
            f"invalid expert indices (out of range [-1, {E})): {invalid_indices[:5]}{'...' if len(invalid_indices) > 5 else ''}"
        )

    out_flat = c_ptr.view(-1, N)

    w_unpacked = torch.empty((N, K), device=device, dtype=torch.float32)
    if has_zp and b_zp_ptr is not None:
        first_expert_scales = b_scale_ptr[0]
        num_groups = first_expert_scales.shape[1]
        zp_unpacked = torch.empty(
            (N, num_groups), device=device, dtype=torch.float32)

    for i in range(num_m_blocks):
        expert_idx = expert_ids_ptr[i].item()

        if expert_idx < -1 or expert_idx >= E:
            expert_idx = -1
        if expert_idx == -1:
            continue

        start_m = i * BLOCK_SIZE_M
        end_m = min(start_m + BLOCK_SIZE_M, num_tokens_post_padded)
        if start_m >= end_m:
            continue

        token_indices = sorted_token_ids_ptr[start_m:end_m]
        valid_mask = token_indices < num_valid_tokens
        valid_token_indices = token_indices[valid_mask]

        if valid_token_indices.numel() == 0:
            continue

        inp_sub = a_ptr[valid_token_indices // top_k]

        w_packed = b_ptr[expert_idx]
        scales = b_scale_ptr[expert_idx]

        if has_zp and b_zp_ptr is not None:
            qzeros = b_zp_ptr[expert_idx]  # [N, K//group_size]
        else:
            qzeros = None

        if use_int4_w4a16:
            w_packed_int32 = w_packed.int()
            w_unpacked[:, 0::2] = (w_packed_int32 & 0xF).float()
            w_unpacked[:, 1::2] = ((w_packed_int32 >> 4) & 0xF).float()
            del w_packed_int32
        elif use_int8_w8a16:
            w_unpacked.copy_(w_packed.float())  # 使用预分配张量
        else:
            raise ValueError(
                f"Unsupported quantization: int4={use_int4_w4a16}, int8={use_int8_w8a16}")

        num_groups = scales.shape[1]

        if has_zp and qzeros is not None:
            if use_int4_w4a16:
                N_packed = qzeros.shape[0]
                zp_int32 = qzeros.int()
                zp_unpacked[:N_packed * 2:2, :] = (zp_int32 & 0xF).float()
                zp_unpacked[1:N_packed * 2:2,
                            :] = ((zp_int32 >> 4) & 0xF).float()
                del zp_int32
            else:
                zp_unpacked.copy_(qzeros.float())
        else:
            if use_int4_w4a16:
                zp = 8.0
            elif use_int8_w8a16:
                zp = 128.0
            else:
                zp = 0.0

        w_reshaped = w_unpacked.view(N, num_groups, group_size)
        scales_reshaped = scales.float().unsqueeze(2)

        if has_zp and qzeros is not None:
            zp_reshaped = zp_unpacked.unsqueeze(2)
            W_dequant = (w_reshaped - zp_reshaped) * scales_reshaped
            del zp_reshaped
        else:
            W_dequant = (w_reshaped - zp) * scales_reshaped

        W_dequant = W_dequant.view(N, K)
        W_dequant = W_dequant.to(dtype)

        del w_reshaped, scales_reshaped

        res = torch.matmul(inp_sub.to(dtype), W_dequant.t()).to(torch.float32)

        if MUL_ROUTED_WEIGHT and topk_weights_ptr is not None:
            flat_weights = topk_weights_ptr.flatten()
            router_weights = flat_weights[valid_token_indices]
            res = res * router_weights[:, None]

        out_flat[valid_token_indices] = res.to(dtype)

    return None


def invoke_fused_moe_kernel(A: torch.Tensor,
                            B: torch.Tensor,
                            C: torch.Tensor,
                            A_scale: Optional[torch.Tensor],
                            B_scale: Optional[torch.Tensor],
                            B_zp: Optional[torch.Tensor],
                            topk_weights: Optional[torch.Tensor],
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool,
                            top_k: int,
                            config: dict[str, Any],
                            compute_type: tl.dtype,
                            use_fp8_w8a8: bool,
                            use_int8_w8a8: bool,
                            use_int8_w8a16: bool,
                            use_int4_w4a16: bool,
                            per_channel_quant: bool,
                            block_shape: Optional[list[int]] = None,
                            B_bias: Optional[torch.Tensor] = None) -> None:
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
        assert (block_shape is None
                or triton.cdiv(B.size(-2), block_shape[0]) == B_scale.size(-2))
        assert (block_shape is None
                or triton.cdiv(B.size(-1), block_shape[1]) == B_scale.size(-1))

    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    M = A.size(0)
    num_tokens = M * top_k

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique,
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.size(0),
                 A.size(0) * top_k * config['BLOCK_SIZE_M'])
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.size(1), META['BLOCK_SIZE_N']), )
    HAS_BIAS = B_bias is not None
    if (use_int8_w8a16 or use_int4_w4a16) and \
            block_shape is not None and block_shape[1] > 0:
        assert B_scale is not None and B_scale.ndim == 3
        assert B_zp is None or B_zp.ndim == 3

        use_moe_wna16_cuda = should_moe_wna16_use_cuda(
            num_valid_tokens=num_tokens,
            group_size=block_shape[1],
            num_experts=B.size(0),
            bit=4 if use_int4_w4a16 else 8)
        config = config.copy()
        config.update(
            get_moe_wna16_block_config(config=config,
                                       use_moe_wna16_cuda=use_moe_wna16_cuda,
                                       num_valid_tokens=num_tokens,
                                       size_k=A.size(1),
                                       size_n=B.size(1),
                                       num_experts=B.size(1),
                                       group_size=block_shape[1],
                                       real_top_k=top_k,
                                       block_size_m=config["BLOCK_SIZE_M"]))

        if use_moe_wna16_cuda:
            bit = 4 if use_int4_w4a16 else 8
            torch.ops.xspeedgate_ops.moe_wna16_gemm(A, C, B, B_scale, B_zp,
                               topk_weights if mul_routed_weight else None,
                               sorted_token_ids, expert_ids,
                               num_tokens_post_padded, top_k,
                               config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                               config["BLOCK_SIZE_K"], bit)
            return

        fused_moe_kernel_gptq_awq_torch(
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.size(1),
            A.size(1),
            EM,
            num_tokens,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            C.stride(2),
            B_scale.stride(0),
            B_scale.stride(2),
            B_scale.stride(1),
            B_zp.stride(0) if B_zp is not None else 0,
            B_zp.stride(2) if B_zp is not None else 0,
            B_zp.stride(1) if B_zp is not None else 0,
            block_k_diviable=A.size(1) % config["BLOCK_SIZE_K"] == 0,
            group_size=block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            has_zp=B_zp is not None,
            use_int4_w4a16=use_int4_w4a16,
            use_int8_w8a16=use_int8_w8a16,
            **config,
        )
    else:
        raise NotImplementedError("Only block quantization with W4A16 and W8A16 is supported for now.")


def should_moe_wna16_use_cuda(num_valid_tokens: int, group_size: int,
                              num_experts: int, bit: int):
    return False and bit == 4 and group_size in [32, 64, 128] and num_valid_tokens / num_experts <= 6


# This is used by the Deepseek-V2 and Deepseek-V3 model
@torch.compile(dynamic=True, backend="aot_eager")
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK and \
            current_platform.is_cuda() and \
            num_expert_group <= 32 and topk <= 32 and \
            e_score_correction_bias is not None:
        return fused_grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            e_score_correction_bias=e_score_correction_bias,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor)

    assert hidden_states.size(0) == gating_output.size(0), (
        "Number of tokens mismatch")

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (scores.view(num_token, num_expert_group,
                                    -1).topk(2, dim=-1)[0].sum(dim=-1))
    else:
        group_scores = scores.view(num_token, num_expert_group,
                                   -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.size(-1) // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(),
                                    float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    use_mxfp4_w4a4: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,  #noqa: UP006
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> None:
    fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids, True,
                       activation, apply_router_weight_on_input, use_fp8_w8a8,
                       use_int8_w8a8, use_int8_w8a16, use_int4_w4a16,
                       use_mxfp4_w4a4, per_channel_quant, global_num_experts,
                       expert_map, w1_scale, w2_scale, w1_zp, w2_zp, a1_scale,
                       a2_scale, block_shape, w1_bias, w2_bias)


direct_register_custom_op(
    op_name="inplace_fused_experts_kunlun",
    op_func=inplace_fused_experts,
    mutates_args=["hidden_states"],
    fake_impl=inplace_fused_experts_fake,
    tags=(() if is_torch_equal_or_newer("2.7.0") else
          (torch.Tag.needs_fixed_stride_order, )),
)


def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    use_mxfp4_w4a4: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,  #noqa: UP006
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids, False, activation,
        apply_router_weight_on_input, use_fp8_w8a8, use_int8_w8a8,
        use_int8_w8a16, use_int4_w4a16, use_mxfp4_w4a4, per_channel_quant,
        global_num_experts, expert_map, w1_scale, w2_scale, w1_zp, w2_zp,
        a1_scale, a2_scale, block_shape, w1_bias, w2_bias)


direct_register_custom_op(
    op_name="outplace_fused_experts_kunlun",
    op_func=outplace_fused_experts,
    fake_impl=outplace_fused_experts_fake,
    tags=(() if is_torch_equal_or_newer("2.7.0") else
          (torch.Tag.needs_fixed_stride_order, )),
)


def torch_vllm_inplace_fused_experts(**kwargs) -> torch.Tensor:
    torch.ops.vllm.inplace_fused_experts_kunlun(**kwargs)
    hidden_states = kwargs['hidden_states']
    return hidden_states


def torch_vllm_outplace_fused_experts(**kwargs) -> torch.Tensor:
    return torch.ops.vllm.outplace_fused_experts_kunlun(**kwargs)


def dispatch_fused_experts_func(inplace: bool) -> Callable[..., torch.Tensor]:
    if inplace:
        return torch_vllm_inplace_fused_experts
    return torch_vllm_outplace_fused_experts


# TODO (bnell): replace this with modular op.  Can get rid of inplace/outplace
# torch ops.
def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    quant_config: Optional[FusedMoEQuantConfig] = None,
    allow_deep_gemm: bool = False,
    allow_cutlass_block_scaled_grouped_gemm: bool = False,
) -> torch.Tensor:

    if quant_config is None:
        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG
    use_fp8_w8a8 = quant_config.use_fp8_w8a8

    # For now, disable DeepGemm for small N (<= 512) until better
    # permute/unpermute ops are available.
    # However, on B200, we use DeepGemm for all cases because they only support
    # E8M0 scale, which means we requantize the weight and input to the specific
    # scale. Fallen back to cutlass or triton for some cases would cause
    # accuracy issue.
    if (allow_deep_gemm and quant_config.use_fp8_w8a8 and
        (is_deep_gemm_e8m0_used() or _valid_deep_gemm(hidden_states, w1, w2))):
        raise NotImplementedError("DeepGemm fused experts is not implemented yet.")
    elif (allow_cutlass_block_scaled_grouped_gemm and use_fp8_w8a8
          and _valid_cutlass_block_scaled_grouped_gemm(
              w1, w2, inplace, activation, apply_router_weight_on_input,
              expert_map)):
        raise NotImplementedError("Cutlass block scaled grouped gemm is not implemented yet.")
    else:
        return dispatch_fused_experts_func(inplace)(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_fp8_w8a8=quant_config.use_fp8_w8a8,
            use_int8_w8a8=quant_config.use_int8_w8a8,
            use_int8_w8a16=quant_config.use_int8_w8a16,
            use_int4_w4a16=quant_config.use_int4_w4a16,
            use_mxfp4_w4a4=quant_config.use_mxfp4_w4a4,
            per_channel_quant=quant_config.per_act_token_quant,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=quant_config.w1_scale,
            w2_scale=quant_config.w2_scale,
            w1_zp=quant_config.w1_zp,
            w2_zp=quant_config.w2_zp,
            a1_scale=quant_config.a1_scale,
            a2_scale=quant_config.a2_scale,
            block_shape=quant_config.block_shape,
            w1_bias=quant_config.w1_bias,
            w2_bias=quant_config.w2_bias)


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    use_mxfp4_w4a4: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.size(1) // 2 == w1.size(2), (
            "Hidden size mismatch")
    elif use_mxfp4_w4a4:
        # 16bit activation and fp4x2 packed weight
        assert hidden_states.size(1) // 2 == w1.size(2), "hidden size mismatch"
    else:
        assert hidden_states.size(1) == w1.size(2), (
            f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}")

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]

    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    K = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)

    config_dtype = _get_config_dtype_str(use_fp8_w8a8=use_fp8_w8a8,
                                         use_int8_w8a16=use_int8_w8a16,
                                         use_int4_w4a16=use_int4_w4a16,
                                         use_mxfp4_w4a4=use_mxfp4_w4a4,
                                         dtype=hidden_states.dtype)

    # Note: for use_int8_w8a16 or use_int4_w4a16, the activations are
    # quantized prior to calling fused_experts.
    quant_dtype = _get_config_quant_dtype(use_fp8_w8a8=use_fp8_w8a8,
                                          use_int8_w8a8=use_int8_w8a8,
                                          use_mxfp4_w4a4=use_mxfp4_w4a4)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.size(),
        w2.size(),
        top_k_num,
        config_dtype,
        block_shape=block_shape,
    )

    config = get_config_func(M)

    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(M * top_k_num * max(N, K),
                          device=hidden_states.device,
                          dtype=hidden_states.dtype)
    intermediate_cache1 = cache13[:M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[:M * top_k_num * K].view(M, top_k_num, K)

    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty((M * top_k_num, N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    if use_mxfp4_w4a4:
        # Weight has to be dequantized for mxfp4 emulation.
        raise NotImplementedError("MXFP4 with W4A4 is not implemented yet.")

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.size()

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk *
                                                      topk_ids.size(1)]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(
            A=curr_hidden_states,
            A_scale=a1_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape)

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'],
                                 global_num_experts, expert_map))

        invoke_fused_moe_kernel(qcurr_hidden_states,
                                w1,
                                intermediate_cache1,
                                a1q_scale,
                                w1_scale,
                                w1_zp,
                                curr_topk_weights,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                apply_router_weight_on_input,
                                top_k_num,
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                B_bias=w1_bias)

        # Activation function with multiplication
        if activation == "silu":
            torch.ops._C.silu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        elif activation == "swigluoai":
            # alpha = 1.702, limit = 7.0
            torch.ops._C.swigluoai_and_mul(intermediate_cache2,
                                           intermediate_cache1.view(-1, N))
        # Activation function without multiplication
        elif activation == SILU_NO_MUL:
            intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))
        elif activation == GELU_NO_MUL:
            intermediate_cache2 = F.gelu(intermediate_cache1.view(-1, N))

        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}.")

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            A=intermediate_cache2,
            A_scale=a2_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape)

        invoke_fused_moe_kernel(qintermediate_cache2,
                                w2,
                                intermediate_cache3,
                                a2q_scale,
                                w2_scale,
                                w2_zp,
                                curr_topk_weights,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                not apply_router_weight_on_input,
                                1,
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                B_bias=w2_bias)

        kunlun_ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.size()),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx])

    return out_hidden_states