# SPDX-License-Identifier: Apache-2.0
"""Bit-preserving native-DCP output/LSE exchange for small decode batches."""

import torch
import torch.distributed as dist
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.dcp.sfa_dcp_merge import fused_merge

_HEAD_DIM = 512
_MAX_LOCAL_HEADS = 32  # Current local-head validation scope.


@triton.jit
def _pack_t1_peer(output_words_ptr, lse_ptr, send_ptr, LOCAL_HEADS: tl.constexpr, HEAD_WORDS: tl.constexpr):
    peer = tl.program_id(0)
    OUTPUT_WORDS: tl.constexpr = LOCAL_HEADS * HEAD_WORDS
    PEER_WORDS: tl.constexpr = OUTPUT_WORDS + LOCAL_HEADS
    word = tl.arange(0, OUTPUT_WORDS)
    values = tl.load(output_words_ptr + peer * OUTPUT_WORDS + word, mask=word < OUTPUT_WORDS, other=0)
    tl.store(send_ptr + peer * PEER_WORDS + word, values, mask=word < OUTPUT_WORDS)
    local_head = tl.arange(0, LOCAL_HEADS)
    stats = tl.load(
        lse_ptr + peer * LOCAL_HEADS + local_head,
        mask=local_head < LOCAL_HEADS,
        other=0.0,
    )
    tl.store(
        send_ptr + peer * PEER_WORDS + OUTPUT_WORDS + local_head,
        stats.to(tl.int32, bitcast=True),
        mask=local_head < LOCAL_HEADS,
    )


@triton.jit(do_not_specialize=["tokens"])
def _pack_token_peer(
    output_words_ptr,
    lse_ptr,
    send_ptr,
    output_stride_token_words,
    output_stride_head_words,
    lse_stride_token,
    lse_stride_head,
    tokens,
    LOCAL_HEADS: tl.constexpr,
    HEAD_WORDS: tl.constexpr,
    STATS_TILE: tl.constexpr,
):
    peer = tl.program_id(0)
    OUTPUT_WORDS: tl.constexpr = LOCAL_HEADS * HEAD_WORDS
    peer_words = tokens * (OUTPUT_WORDS + LOCAL_HEADS)
    word = tl.arange(0, OUTPUT_WORDS)
    local_head = word // HEAD_WORDS
    dim_word = word % HEAD_WORDS
    peer_base = peer * peer_words
    for token in range(tokens):
        values = tl.load(
            output_words_ptr
            + token * output_stride_token_words
            + (peer * LOCAL_HEADS + local_head) * output_stride_head_words
            + dim_word
        )
        tl.store(send_ptr + peer_base + token * OUTPUT_WORDS + word, values)
    index = tl.arange(0, STATS_TILE)
    token, head = index // LOCAL_HEADS, index % LOCAL_HEADS
    stats = tl.load(
        lse_ptr + token * lse_stride_token + (peer * LOCAL_HEADS + head) * lse_stride_head,
        mask=index < tokens * LOCAL_HEADS,
        other=0.0,
    )
    tl.store(
        send_ptr + peer_base + tokens * OUTPUT_WORDS + index,
        stats.to(tl.int32, bitcast=True),
        mask=index < tokens * LOCAL_HEADS,
    )


def can_exchange(output: torch.Tensor, lse: torch.Tensor, ranks: int = 8) -> bool:
    # These properties are uniform within a valid DCP group. Do not choose
    # different collective counts based on rank-local allocation/stride state.
    return (
        output.dtype == torch.bfloat16
        and output.ndim == 3
        and ranks > 1
        and ranks & (ranks - 1) == 0  # Merge's tl.arange(0, RANKS) requires a power-of-two length.
        and output.shape[0] > 0  # Tests configured at most 192 tokens; this is not a kernel limit.
        and output.shape[2] == _HEAD_DIM
        and output.shape[1] % ranks == 0
        and 0 < output.shape[1] // ranks <= _MAX_LOCAL_HEADS
        # Full vectors preserve the unmasked copy lowering. Other head counts
        # retain upstream packing rather than scalarizing a masked token loop.
        and ((output.shape[1] // ranks) & ((output.shape[1] // ranks) - 1)) == 0
        and lse.dtype == torch.float32
        and tuple(lse.shape) == (output.shape[0], output.shape[1], 1)
    )


def _exchange_tokens(output: torch.Tensor, lse: torch.Tensor, group, ranks: int) -> torch.Tensor:
    tokens = output.shape[0]
    head_dim = output.shape[-1]
    local_heads = output.shape[1] // ranks
    head_words = head_dim // 2  # Two BF16 values per INT32 word.
    output_words = local_heads * head_words
    # Stride-dependent normalization stays inside the one-collective path.
    if output.stride(-1) != 1 or any(stride % 2 for stride in output.stride()[:2]):
        output = output.contiguous()
    if output.storage_offset() % 2:
        output = output.clone()
    words = output.view(torch.int32)
    peer_words = tokens * (output_words + local_heads)
    send = torch.empty((ranks, peer_words), dtype=torch.int32, device=output.device)
    _pack_token_peer[(ranks,)](
        words,
        lse,
        send,
        words.stride(0),
        words.stride(1),
        lse.stride(0),
        lse.stride(1),
        tokens,
        local_heads,
        head_words,
        triton.next_power_of_2(tokens * local_heads),  # Power-of-two LSE vector; this constexpr varies with T.
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    # Final receiver layout is token-major: no transpose or receive-side copy.
    parts = recv.view(output.dtype).as_strided(
        (ranks, tokens, local_heads, head_dim),
        (peer_words * 2, local_heads * head_dim, head_dim, 1),
    )
    stats = recv.view(torch.float32).as_strided(
        (ranks, tokens, local_heads),
        (peer_words, local_heads, 1),
        storage_offset=tokens * output_words,
    )
    return fused_merge(parts, stats, token_dim=1)


def exchange(output: torch.Tensor, lse: torch.Tensor, group) -> torch.Tensor:
    """Exchange head shards, retaining FP32 LSE bits and merge math."""
    ranks = dist.get_world_size(group)
    if output.shape[0] != 1:
        return _exchange_tokens(output, lse, group, ranks)
    head_dim = output.shape[-1]
    local_heads = output.shape[1] // ranks
    head_words = head_dim // 2  # Two BF16 values per INT32 word.
    output_words = local_heads * head_words
    peer_words = output_words + local_heads
    output = output.contiguous()
    if output.storage_offset() % 2:
        output = output.clone()
    lse = lse.contiguous()
    words = output.view(torch.int32)
    send = torch.empty((ranks, peer_words), dtype=torch.int32, device=output.device)
    # T1 normalizes LSE to contiguous [1, heads, 1], so the head stride is one.
    _pack_t1_peer[(ranks,)](words, lse, send, local_heads, head_words)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    parts = recv.view(output.dtype).as_strided(
        (ranks, local_heads, 1, head_dim),
        (peer_words * 2, head_dim, head_dim, 1),
    )
    stats = recv.view(torch.float32).as_strided(
        (ranks, local_heads, 1),
        (peer_words, 1, 1),
        storage_offset=output_words,
    )
    return fused_merge(parts, stats, token_dim=2)
