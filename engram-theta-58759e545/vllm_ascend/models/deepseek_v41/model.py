# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 text model and source-shared hybrid-cache graph."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import vllm.envs as envs
from safetensors import safe_open
from transformers import AutoTokenizer
from vllm.config import CUDAGraphMode
from vllm.distributed import (
    get_dp_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import get_forward_context, is_forward_context_available

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheBackend,
    DeepseekV41CacheLayer,
    DeepseekV41Metadata,
)
from vllm_ascend.core.deepseek_v41 import (
    DeepseekV41FullSpec,
    DeepseekV41SWASpec,
    validate_cache_runtime,
)
from vllm_ascend.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm_ascend.models.deepseek_v4 import (
    AscendDeepseekV4ForCausalLM,
    AscendDeepseekV4SWACache,
    DeepseekV2DecoderLayer,
    DeepseekV4Attention,
    DeepseekV4Model,
)

from .compressor import DeepseekV41Compressor, _read, text_config_of
from .engram_gate import engram_gate
from .engram_hash import PagedNgramHistory, engram_history_metadata
from .engram_hbm import EngramQueryGroup, NodeShardedEngram
from .indexer import DeepseekV41Indexer


@dataclass(frozen=True)
class DeepseekV41LayerRole:
    """The attention and future Engram responsibilities of one backbone layer."""

    layer_idx: int
    compress_ratio: int
    kv_source_layer: int | None
    index_source_layer: int | None
    is_kv_source: bool
    is_index_source: bool
    is_candidate_source: bool
    uses_candidate_filter: bool
    engram_slot: int | None

    @property
    def has_long_context(self) -> bool:
        return self.compress_ratio > 0


@dataclass(frozen=True)
class DeepseekV41Topology:
    """Validated, immutable model-wide source/consumer topology."""

    layers: tuple[DeepseekV41LayerRole, ...]
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    candidate_source_layer: int
    candidate_topk_blocks: int
    candidate_block_size: int
    index_topk: int

    def layer(self, layer_idx: int) -> DeepseekV41LayerRole:
        return self.layers[layer_idx]

    def kv_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(role.layer_idx for role in self.layers if role.kv_source_layer == source_layer)

    def index_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(role.layer_idx for role in self.layers if role.index_source_layer == source_layer)


class DeepseekV41SharedAttentionState:
    """Per-forward handoff between index sources and their consumer layers."""

    def __init__(self, topk_indices, candidates):
        self.topk_indices = topk_indices
        self.candidates = candidates

    def reset(self):
        # Source layers overwrite the active rows before any consumer reads
        # them. Keeping the storage intact avoids replay depending on Python
        # state mutation and preserves a fixed address for ACL Graph.
        return None


@dataclass(frozen=True)
class DeepseekV41DecoderTailPlan:
    """Token rows retained after the shared encoder/candidate layer."""

    token_indices_cpu: torch.Tensor
    query_start_loc_cpu: torch.Tensor


def build_decoder_tail_plan(
    metadata: DeepseekV41Metadata,
    window_size: int = 128,
    num_computed_tokens_cpu: torch.Tensor | None = None,
    num_prompt_tokens_cpu: torch.Tensor | None = None,
) -> DeepseekV41DecoderTailPlan:
    """Intersect each prefill chunk with the prompt's global tail window."""
    if window_size <= 0:
        raise ValueError("DeepSeek V4.1 decoder tail window must be positive")
    starts = metadata.query_start_loc_cpu
    flags = metadata.is_prefilling
    if starts is None or flags is None:
        raise RuntimeError("DeepSeek V4.1 eager decoder tail requires CPU request metadata")
    if starts.device.type != "cpu" or flags.device.type != "cpu":
        raise RuntimeError("DeepSeek V4.1 eager decoder tail request metadata must stay on CPU")

    num_reqs = metadata.num_reqs
    has_global_positions = num_computed_tokens_cpu is not None and num_prompt_tokens_cpu is not None
    if (num_computed_tokens_cpu is None) != (num_prompt_tokens_cpu is None):
        raise ValueError("DeepSeek V4.1 eager decoder tail requires both computed and prompt lengths")
    if has_global_positions:
        assert num_computed_tokens_cpu is not None
        assert num_prompt_tokens_cpu is not None
        if num_computed_tokens_cpu.device.type != "cpu" or num_prompt_tokens_cpu.device.type != "cpu":
            raise RuntimeError("DeepSeek V4.1 eager decoder tail global positions must stay on CPU")
        if num_computed_tokens_cpu.numel() < num_reqs or num_prompt_tokens_cpu.numel() < num_reqs:
            raise ValueError("DeepSeek V4.1 eager decoder tail global positions are shorter than the request batch")
    starts = starts[: num_reqs + 1].long()
    flags = flags[:num_reqs].bool()
    pieces = []
    lengths = []
    for request_idx in range(num_reqs):
        start = int(starts[request_idx])
        end = int(starts[request_idx + 1])
        is_prefill = bool(flags[request_idx])
        retained_end = end
        if is_prefill and has_global_positions:
            assert num_computed_tokens_cpu is not None
            assert num_prompt_tokens_cpu is not None
            chunk_start = int(num_computed_tokens_cpu[request_idx])
            prompt_end = int(num_prompt_tokens_cpu[request_idx])
            query_len = end - start
            tail_start = max(0, prompt_end - window_size)
            local_start = min(query_len, max(0, tail_start - chunk_start))
            local_end = min(query_len, max(0, prompt_end - chunk_start))
            retained_start = start + local_start
            retained_end = start + max(local_start, local_end)
        else:
            retained_start = max(start, end - window_size) if is_prefill else start
        pieces.append(torch.arange(retained_start, retained_end, dtype=torch.long))
        lengths.append(retained_end - retained_start)

    indices = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=metadata.query_start_loc_cpu.dtype)
    if lengths:
        query_start_loc[1:] = torch.tensor(lengths, dtype=query_start_loc.dtype).cumsum(0)
    return DeepseekV41DecoderTailPlan(indices, query_start_loc)


def build_decoder_tail_metadata(
    metadata: dict[str, Any],
    plan: DeepseekV41DecoderTailPlan,
) -> dict[str, Any]:
    """Re-index V4.1 cache metadata for the eager decoder-tail batch."""
    representative = next(
        (value for value in metadata.values() if isinstance(value, DeepseekV41Metadata)),
        None,
    )
    if representative is None:
        return metadata

    num_reqs = representative.num_reqs
    query_lens = plan.query_start_loc_cpu[1:] - plan.query_start_loc_cpu[:-1]
    flags = representative.is_prefilling[:num_reqs].bool()
    num_prefills = int(flags.sum().item())
    num_decodes = num_reqs - num_prefills
    num_prefill_tokens = int(query_lens[flags].sum().item())
    num_decode_tokens = int(query_lens[~flags].sum().item())
    num_actual_reqs = min(representative.num_actual_reqs, num_reqs)
    num_actual_tokens = int(plan.query_start_loc_cpu[num_actual_reqs])
    num_input_tokens = int(plan.token_indices_cpu.numel())
    max_query_len = int(query_lens.max().item()) if num_reqs else 0

    tensor_cache: dict[int, torch.Tensor] = {}
    rope_cache: dict[int, Any] = {}

    def slice_tensor(value: torch.Tensor | None):
        if value is None:
            return None
        cached = tensor_cache.get(id(value))
        if cached is None:
            indices = plan.token_indices_cpu.to(value.device)
            cached = value.index_select(0, indices)
            tensor_cache[id(value)] = cached
        return cached

    def slice_rope(value):
        if value is None:
            return None
        cached = rope_cache.get(id(value))
        if cached is None:
            if hasattr(value, "items"):
                cached = {name: slice_tensor(tensor) for name, tensor in value.items()}
            else:
                indices = plan.token_indices_cpu.to(representative.query_start_loc.device)
                cached = value[indices]
            rope_cache[id(value)] = cached
        return cached

    result = {}
    for name, value in metadata.items():
        if not isinstance(value, DeepseekV41Metadata):
            result[name] = value
            continue
        result[name] = replace(
            value,
            query_start_loc=plan.query_start_loc_cpu.to(value.query_start_loc.device),
            query_start_loc_cpu=plan.query_start_loc_cpu,
            slot_mapping=slice_tensor(value.slot_mapping),
            positions=slice_tensor(value.positions),
            cos=slice_rope(value.cos),
            sin=slice_rope(value.sin),
            num_actual_tokens=num_actual_tokens,
            num_input_tokens=num_input_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            max_query_len=max_query_len,
            smla_metadata=None,
            qli_metadata=None,
            rebuild_operator_metadata=True,
        )
    return result


def _as_int_tuple(config: Any, name: str) -> tuple[int, ...]:
    value = _read(config, name)
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, int) for item in value):
        raise ValueError(f"DeepSeek V4.1 {name} must be a list of integers")
    return tuple(value)


def _latest_source(layer_idx: int, sources: tuple[int, ...]) -> int | None:
    return next((source for source in reversed(sources) if source <= layer_idx), None)


def decoder_tail_work_rows(local_tail_tokens: int, participating: bool, tp_size: int, flash_comm: bool) -> int:
    """Rows this rank feeds to the late layers for one eager forward.

    ``participating`` is the DP-wide decision: at least one rank holds rows
    inside the global window, so the late layers must run everywhere. A
    participating rank without rows of its own runs a single padded row
    ("陪跑") so it still joins the tail collectives; with flashcomm1 the
    DP-local row count must additionally be a multiple of the TP size, because
    the tail sequence is sharded across TP.

    Returns 0 only when no DP rank has tail work, i.e. the late layers can be
    skipped on every rank.
    """
    if not participating:
        return 0
    rows = max(local_tail_tokens, 1)
    if flash_comm and tp_size > 1:
        rows = -(-rows // tp_size) * tp_size
    return rows


def any_dp_rank_has_tail(local_tail_tokens: int) -> bool:
    """DP-wide OR of "this rank has rows inside the global window".

    Every DP rank must call this in the same forward: the decision has to be
    identical everywhere, otherwise the tail collectives desynchronise. The
    flag travels on the CPU (gloo) group, matching the DP token-count all-reduce
    the runner already performs.
    """
    dp_group = get_dp_group()
    if dp_group.world_size == 1:
        return local_tail_tokens > 0
    flag = torch.zeros(1, dtype=torch.int32, device="cpu")
    flag[0] = 1 if local_tail_tokens else 0
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=dp_group.cpu_group)
    return bool(flag[0].item())


def extend_decoder_tail_plan_with_pads(plan: DeepseekV41DecoderTailPlan, pad_rows: int) -> DeepseekV41DecoderTailPlan:
    """Append ``pad_rows`` throwaway rows to a tail plan.

    The pads are appended after every real row, i.e. they belong to the span of
    the *last* request in the flattened batch, so only the final boundary of
    ``query_start_loc`` grows. Their inputs copy the last real row (keeping the
    row's positions consistent with the span they land in) and their cache slots
    are blanked afterwards, so they only exist to keep the per-rank row count
    nonzero / TP-aligned.
    """
    if pad_rows <= 0:
        return plan
    indices = plan.token_indices_cpu
    pad_index = indices[-1] if indices.numel() else torch.zeros((), dtype=torch.long)
    pads = pad_index.reshape(1).expand(pad_rows).clone()
    padded = torch.cat([indices, pads])
    query_start_loc = plan.query_start_loc_cpu.clone()
    if query_start_loc.numel() > 1:
        query_start_loc[-1] += pad_rows
    return DeepseekV41DecoderTailPlan(padded, query_start_loc)


def blank_padded_slots(metadata: dict[str, Any], first_pad_row: int) -> dict[str, Any]:
    """Point every padded row's cache slot at -1 so it writes nothing.

    ``-1`` is the no-write sentinel the dummy capture path already uses
    (``slot_mapping.fill_(-1)``).
    """
    result = {}
    for name, value in metadata.items():
        if not isinstance(value, DeepseekV41Metadata) or value.slot_mapping is None:
            result[name] = value
            continue
        if value.slot_mapping.shape[0] <= first_pad_row:
            result[name] = value
            continue
        slots = value.slot_mapping.clone()
        slots[first_pad_row:] = -1
        result[name] = replace(value, slot_mapping=slots)
    return result


def build_layer_plan(config: Any) -> DeepseekV41Topology:
    """Build and validate the V4.1 layer-sharing graph from a text config.

    ``config`` may be a Transformers config object or the raw ``text_config``
    dictionary.  Extra compression ratios for speculative layers are allowed,
    but only the first ``num_hidden_layers`` entries describe the backbone.
    """

    config = text_config_of(config)
    num_layers = int(_read(config, "num_hidden_layers"))
    ratios = _as_int_tuple(config, "compress_ratios")
    kv_sources = _as_int_tuple(config, "kv_source_layers")
    index_sources = _as_int_tuple(config, "index_source_layers")
    engram_layers = _as_int_tuple(config, "engram_layer_ids")
    candidate_source = int(_read(config, "candidate_source_layer"))
    candidate_topk_blocks = int(_read(config, "candidate_topk_blocks"))
    candidate_block_size = int(_read(config, "candidate_block_size"))
    index_topk = int(_read(config, "index_topk"))

    if num_layers <= 0:
        raise ValueError("DeepSeek V4.1 num_hidden_layers must be positive")
    if len(ratios) < num_layers:
        raise ValueError(
            "DeepSeek V4.1 compress_ratios must cover every backbone layer: "
            f"got {len(ratios)} ratios for {num_layers} layers"
        )
    ratios = ratios[:num_layers]
    if any(ratio not in (0, 1, 2) for ratio in ratios):
        raise ValueError(f"DeepSeek V4.1 backbone only supports compression ratios 0, 1 and 2; got {ratios}")

    for name, sources in (("kv_source_layers", kv_sources), ("index_source_layers", index_sources)):
        if tuple(sorted(set(sources))) != sources:
            raise ValueError(f"DeepSeek V4.1 {name} must be sorted and unique")
        if any(source < 0 or source >= num_layers for source in sources):
            raise ValueError(f"DeepSeek V4.1 {name} contains a layer outside the backbone")
        if any(ratios[source] == 0 for source in sources):
            raise ValueError(f"DeepSeek V4.1 {name} cannot point to a local-only layer")

    if not set(kv_sources).issubset(index_sources):
        raise ValueError("Every DeepSeek V4.1 KV source must also be an index source")
    if candidate_source not in kv_sources:
        raise ValueError("DeepSeek V4.1 candidate_source_layer must be a KV source")
    if candidate_topk_blocks <= 0 or candidate_block_size <= 0 or index_topk <= 0:
        raise ValueError("DeepSeek V4.1 candidate and index TopK values must be positive")
    if len(set(engram_layers)) != len(engram_layers):
        raise ValueError("DeepSeek V4.1 engram_layer_ids must be unique")
    if any(layer < 0 or layer >= num_layers for layer in engram_layers):
        raise ValueError("DeepSeek V4.1 engram_layer_ids contains a layer outside the backbone")

    engram_slots = {layer_idx: slot for slot, layer_idx in enumerate(engram_layers)}
    roles: list[DeepseekV41LayerRole] = []
    for layer_idx, ratio in enumerate(ratios):
        kv_source = _latest_source(layer_idx, kv_sources) if ratio else None
        index_source = _latest_source(layer_idx, index_sources) if ratio else None
        if ratio and (kv_source is None or index_source is None):
            raise ValueError(f"DeepSeek V4.1 layer {layer_idx} has long-context attention but no source layer")
        if kv_source is not None and ratios[kv_source] != ratio:
            raise ValueError(
                f"DeepSeek V4.1 layer {layer_idx} has ratio {ratio}, but its KV source "
                f"layer {kv_source} has ratio {ratios[kv_source]}"
            )

        roles.append(
            DeepseekV41LayerRole(
                layer_idx=layer_idx,
                compress_ratio=ratio,
                kv_source_layer=kv_source,
                index_source_layer=index_source,
                is_kv_source=layer_idx in kv_sources,
                is_index_source=layer_idx in index_sources,
                is_candidate_source=layer_idx == candidate_source,
                # Consumer layers inherit the selection policy of their index
                # source.  For example, layer 26 reuses layer 24 TopK, and that
                # TopK was computed inside layer 20's candidate blocks.
                uses_candidate_filter=index_source is not None and index_source > candidate_source,
                engram_slot=engram_slots.get(layer_idx),
            )
        )

    return DeepseekV41Topology(
        layers=tuple(roles),
        kv_source_layers=kv_sources,
        index_source_layers=index_sources,
        candidate_source_layer=candidate_source,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        index_topk=index_topk,
    )


class AscendDeepseekV41SWACache(AscendDeepseekV4SWACache):
    """V4 execution-compatible SWA plane participating in V4.1 grouping."""

    def get_kv_cache_spec(self, vllm_config):
        spec = super().get_kv_cache_spec(vllm_config)
        return DeepseekV41SWASpec(
            block_size=spec.block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            dtype=spec.dtype,
            sliding_window=spec.sliding_window,
            cache_dtype_str=spec.cache_dtype_str,
            model_version="deepseek_v4",
            alignment=spec.alignment,
        )

    def get_attn_backend(self):
        return DeepseekV41CacheBackend


class DeepseekV41Attention(DeepseekV4Attention):
    """V4.1 source-shared attention using V4 projections and CP adapters."""

    swa_cache_cls = AscendDeepseekV41SWACache

    def __init__(
        self,
        vllm_config,
        config,
        max_position_embeddings=0,
        cache_config=None,
        quant_config=None,
        prefix="",
        topk_indices_buffer=None,
    ):
        config = text_config_of(config)
        validate_cache_runtime(vllm_config)
        layer_idx = int(prefix.split(".")[-2])
        topology = build_layer_plan(config)
        role = topology.layer(layer_idx)
        # Reuse V4's quant-aware projections and stable SWA eager backend.  A
        # zero ratio prevents V4 from creating its incompatible c4/c128 planes.
        original_ratios = config.compress_ratios
        config.compress_ratios = tuple(0 for _ in original_ratios)
        try:
            super().__init__(
                vllm_config=vllm_config,
                config=config,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            )
        finally:
            config.compress_ratios = original_ratios
        from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding

        # V4.1 applies YaRN only to layers carrying long-context compressed KV.
        # Pure SWA layers use the unscaled base RoPE even though the allocated
        # lookup table still spans the configured maximum context length.
        self.rotary_emb = ComplexExpRotaryEmbedding(
            vllm_config=vllm_config,
            layername=f"{prefix}.attn",
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            is_neox_style=False,
            scaling_factor=config.rope_parameters["factor"],
            base=(config.compress_rope_theta if role.has_long_context else config.rope_theta),
            beta_fast=config.rope_parameters["beta_fast"],
            beta_slow=config.rope_parameters["beta_slow"],
            original_seq_len=(max_position_embeddings if role.has_long_context else 0),
            rope_groups=["default"],
        )
        block_size = vllm_config.cache_config.block_size
        if block_size <= 0 or block_size % 2:
            raise ValueError("V4.1 logical block_size must be a positive multiple of two")
        owned = []
        if role.is_kv_source:
            owned.extend((f"{prefix}.long_kv_cache", f"{prefix}.indexer.k_cache"))
            if role.compress_ratio == 2:
                owned.append(f"{prefix}.compressor.state_cache")
        duplicates = set(owned) & vllm_config.compilation_config.static_forward_context.keys()
        if duplicates:
            raise ValueError(f"Duplicate V4.1 cache prefixes: {sorted(duplicates)}")
        self.role = role
        self.topology = topology
        self.shared_state = None
        self.prefix = prefix
        width = _read(config, "head_dim")
        self.softmax_scale = width**-0.5
        if role.is_kv_source:
            self.long_kv_cache = DeepseekV41CacheLayer(
                vllm_config,
                f"{prefix}.long_kv_cache",
                DeepseekV41FullSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=width,
                    dtype=torch.bfloat16,
                    compress_ratio=role.compress_ratio,
                ),
            )
        self.compressor = (
            DeepseekV41Compressor(config, role.compress_ratio, vllm_config, f"{prefix}.compressor")
            if role.is_kv_source
            else None
        )
        self.indexer = (
            DeepseekV41Indexer(
                config,
                role.is_kv_source,
                vllm_config,
                f"{prefix}.indexer",
                role.compress_ratio,
                quant_config=quant_config,
            )
            if role.is_index_source
            else None
        )
        root = prefix.rsplit(".layers.", 1)[0]
        source = f"{root}.layers.{role.kv_source_layer}.self_attn"
        self.long_kv_source_prefix = f"{source}.long_kv_cache" if role.has_long_context else None
        self.index_k_source_prefix = f"{source}.indexer.k_cache" if role.has_long_context else None
        self.index_source_layer = role.index_source_layer
        from vllm_ascend.attention.context_parallel.dsa_v41_cp import get_v41_cp_classes

        self.v41_impl = get_v41_cp_classes()[1](
            prefix=prefix,
            role=role,
            topology=topology,
            long_kv_source_prefix=self.long_kv_source_prefix,
            index_k_source_prefix=self.index_k_source_prefix,
        )
        self.v41_layer_name = f"{prefix}.v41_attn"
        context = vllm_config.compilation_config.static_forward_context
        if self.v41_layer_name in context:
            raise ValueError(f"Duplicate V4.1 attention layer: {self.v41_layer_name}")
        context[self.v41_layer_name] = self

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        output = torch.empty_like(hidden_states)
        torch.ops.vllm.dsa_v41_forward(hidden_states, output, self.v41_layer_name)
        return output


class DeepseekV41DecoderLayer(DeepseekV2DecoderLayer):
    """V4.1 block with the checkpoint's delayed mHC coefficient handoff."""

    attention_cls = DeepseekV41Attention

    def __init__(self, vllm_config, prefix, **kwargs):
        super().__init__(vllm_config, prefix, **kwargs)
        self.use_sequence_parallel = vllm_config.parallel_config.use_sequence_parallel_moe
        # Leave the TP partial sums for the reduce-scatter below. The mHC
        # and MoE paths then stay sharded between attention calls.
        if self.use_sequence_parallel:
            self.self_attn.wo_b.reduce_results = False
        config = vllm_config.model_config.hf_config
        engram_enabled = get_ascend_config().enable_engram
        if engram_enabled and self.layer_idx in config.engram_layer_ids:
            self.engram = torch.nn.Module()
            self.engram.wkv = torch.nn.Linear(
                (config.engram_max_ngram_size - 1) * config.engram_n_heads * config.engram_head_dim,
                (config.hc_mult + 1) * config.hidden_size,
                bias=False,
                dtype=torch.bfloat16,
            )
            self.engram.q_weight = torch.nn.Parameter(
                torch.empty(config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
            )
            self.engram.k_weight = torch.nn.Parameter(
                torch.empty(config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
            )
        else:
            self.engram = None

    @staticmethod
    def hc_collapse(x, pre_mix):
        return (pre_mix.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)

    def hc_pre(self, x, hc_fn, hc_scale, hc_base, pre_mix=None):
        return torch.ops._C_ascend.npu_hc_pre_v2(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            pre_mix,
            hc_mult=self.hc_mult,
            hc_sinkhorn_iters=self.hc_sinkhorn_iters,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
        )

    def hc_post(self, x, residual, post, comb):
        return torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(0),
            residual.unsqueeze(0),
            post.unsqueeze(0),
            comb.unsqueeze(0),
        ).squeeze(0)

    def forward(
        self,
        positions,
        hidden_states,
        pre_mix,
        llama_4_scaling=None,
        input_ids=None,
    ):
        use_sequence_parallel = getattr(self, "use_sequence_parallel", False)
        residual = hidden_states
        x, attn_post, attn_comb, attn_pre = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            pre_mix,
        )
        x = self.input_layernorm(x)
        if use_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]
        x = self.self_attn(positions, x, llama_4_scaling)
        if use_sequence_parallel:
            x = sp_reduce_scatter(x)
        hidden_states = self.hc_post(x, residual, attn_post, attn_comb)

        residual = hidden_states
        x, ffn_post, ffn_comb, ffn_pre = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            attn_pre,
        )
        x, x_fp32 = self.rms_norm_cast(x)
        x = self.mlp(
            x,
            input_ids=input_ids,
            hidden_states_fp32=x_fp32,
            already_sequence_parallel=use_sequence_parallel,
        )
        hidden_states = self.hc_post(x, residual, ffn_post, ffn_comb)
        return hidden_states, ffn_pre


class DeepseekV41Model(DeepseekV4Model):
    """Single V4.1 backbone entry, matching ``deepseek_v4/model.py``."""

    decoder_layer_cls = DeepseekV41DecoderLayer

    def __init__(self, *, vllm_config, prefix=""):
        if (
            get_ascend_config().enable_engram
            and vllm_config.load_config.load_format != "dummy"
            and vllm_config.load_config.safetensors_load_strategy != "lazy"
        ):
            raise ValueError("Engram HBM shards require --safetensors-load-strategy lazy")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.vllm_config = vllm_config
        self.use_sequence_parallel = vllm_config.parallel_config.use_sequence_parallel_moe
        # V4.1 collapses with the last block's ffn_pre; it has no hc_head
        # projection in the checkpoint.
        del self.hc_head_fn, self.hc_head_base, self.hc_head_scale
        topology = build_layer_plan(self.config)
        self.decoder_tail_start_layer = topology.candidate_source_layer + 1
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        candidate_buffer = torch.full(
            (max_tokens, 1, topology.candidate_topk_blocks),
            -1,
            dtype=torch.int32,
            device=self.topk_indices_buffer.device,
        )
        self.candidate_indices_buffer = candidate_buffer
        self.shared_attention_state = DeepseekV41SharedAttentionState(
            self.topk_indices_buffer,
            candidate_buffer,
        )
        for layer in self.layers:
            if isinstance(layer, DeepseekV41DecoderLayer):
                layer.self_attn.shared_state = self.shared_attention_state
        config = self.config
        # Target storage is a loader/runtime choice.  Checkpoint metadata is
        # used only by load_checkpoint to validate the source representation.
        # Read the storage choice after AscendConfig validation.
        ascend_config = get_ascend_config()
        self.engram_root = vllm_config.model_config.model
        self.engram_weight_root = ascend_config.engram_model_path or self.engram_root
        storage_format = ascend_config.engram_storage
        if ascend_config.enable_engram:
            query_group = EngramQueryGroup.from_vllm(vllm_config.parallel_config)
            for layer_id, rows in zip(config.engram_layer_ids, config.engram_num_embeddings):
                self.layers[layer_id].engram.embed = NodeShardedEngram(
                    rows,
                    config.engram_head_dim,
                    query_group,
                    storage_format=storage_format,
                    cpu_offload=ascend_config.enable_engram_ple_offload,
                )
        self.engram_history = None
        self._engram_input_buffers = None
        self._engram_max_tokens = max(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )
        self.register_buffer("engram_rotation", torch.eye(32), persistent=False)
        if ascend_config.enable_engram and vllm_config.load_config.load_format != "dummy":
            with torch.device("cpu"):
                tokenizer = AutoTokenizer.from_pretrained(self.engram_root)
                self.engram_history = PagedNgramHistory(config, tokenizer)
                with safe_open(Path(self.engram_root) / "optional/quarot.safetensors", framework="pt") as file:
                    rotation = file.get_tensor("global_rotation")
                block = rotation[:32, :32].contiguous()
                if not torch.equal(rotation, torch.block_diag(*[block] * (config.hidden_size // 32))):
                    raise ValueError("Engram gate requires repeated block32 global rotation")
            self.engram_rotation.copy_(block)

    def prepare_engram(self, input_ids, positions):
        """Eager boundary: every DP participates, including metadata-free dummies."""
        config = self.config
        if not get_ascend_config().enable_engram:
            return {}, torch.empty(0, dtype=torch.bool, device=positions.device)
        columns = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        hashes = torch.empty((0, len(config.engram_layer_ids), columns), dtype=torch.int64, device="cpu")
        mask = torch.empty(0, dtype=torch.bool, device="cpu")
        metadata = get_forward_context().attn_metadata
        if metadata is not None and self.engram_history is not None:
            first = self.layers[0].self_attn.dsa_attn.swa_cache_layer
            meta = metadata[first.prefix]
            boundaries, block_table, block_size = engram_history_metadata(meta)
            n = int(boundaries[-1])
            requests = torch.repeat_interleave(torch.arange(len(boundaries) - 1, device="cpu"), boundaries.diff())
            hashes, mask = self.engram_history.update(
                input_ids[:n].cpu().long(),
                positions[:n].cpu().long(),
                requests,
                block_table,
                block_size,
            )
        lookups = {}
        tables = [self.layers[layer_id].engram.embed for layer_id in config.engram_layer_ids]
        ids_list = [hashes[:, slot] for slot in range(len(tables))]
        if hasattr(tables[0], "route_many"):
            routed = tables[0].route_many(tables, ids_list)
        else:
            routed = [table(ids) for table, ids in zip(tables, ids_list)]
        for layer_id, values in zip(config.engram_layer_ids, routed):
            lookups[layer_id] = values.flatten(1)
        return lookups, mask.to(positions.device)

    def prepare_engram_inputs(self, input_ids, positions, padded_tokens=None):
        """Refresh persistent inputs before main-model capture or replay."""
        lookups, mask = self.prepare_engram(input_ids, positions)
        num_tokens = positions.shape[0]
        # The compiled V4.1 backbone uses the scheduler's static token
        # capacity for decode graphs (typically max_num_batched_tokens), even
        # when the current request has one token.  Keep lookup tensors at that
        # capacity so every captured graph sees the same Engram shape.
        output_tokens = max(self._engram_max_tokens, padded_tokens or 0)
        if output_tokens < num_tokens:
            raise ValueError("Engram padded token count is smaller than the input")
        if self._engram_input_buffers is None:
            capacity = self._engram_max_tokens
            self._engram_input_buffers = (
                {layer: values.new_zeros((capacity, values.shape[1])) for layer, values in lookups.items()},
                mask.new_zeros(capacity),
            )
        buffers, mask_buffer = self._engram_input_buffers
        padded_mask = mask_buffer[:output_tokens]
        padded_mask.zero_()
        padded_mask[: mask.numel()].copy_(mask)
        padded_lookups = {}
        for layer, values in lookups.items():
            padded = buffers[layer][:output_tokens]
            padded.zero_()
            padded[: values.shape[0]].copy_(values)
            padded_lookups[layer] = padded
        return {"engram_lookups": padded_lookups, "engram_mask": padded_mask}

    def _forward_layers(
        self,
        layers,
        positions,
        hidden_states,
        pre_mix,
        input_ids,
        lookups,
        token_mask,
        num_tokens,
    ):
        aux_hidden_states = []
        last_layer = None
        for layer in layers:
            last_layer = layer
            # DSpark consumes the residual stream entering its configured
            # target layers. The runner expresses checkpoint IDs as one-based.
            if layer.layer_idx + 1 in self.aux_hidden_state_layers:
                aux_hidden_state = hidden_states.mean(dim=1)
                if self.use_sequence_parallel:
                    aux_hidden_state = sp_all_gather(aux_hidden_state)[:num_tokens]
                aux_hidden_states.append(aux_hidden_state)
            if layer.engram is not None and token_mask.numel():
                n = hidden_states.shape[0]
                lookup = lookups[layer.layer_idx][:n]
                active_mask = token_mask[:n]
                kv = layer.engram.wkv(lookup)
                key, value = kv.split([self.hc_mult * self.config.hidden_size, self.config.hidden_size], -1)
                hidden_states[:n] = engram_gate(
                    hidden_states[:n],
                    key.view(n, self.hc_mult, self.config.hidden_size),
                    value,
                    layer.engram.q_weight.float() * layer.engram.k_weight.float(),
                    self.engram_rotation,
                    active_mask,
                    self.config.rms_norm_eps,
                )
            hidden_states, pre_mix = layer(positions, hidden_states, pre_mix, None, input_ids=input_ids)
        return hidden_states, pre_mix, last_layer, aux_hidden_states

    @staticmethod
    def _gather_hc_rows(hidden_states, num_tokens, hc_mult, hidden_size):
        gathered = sp_all_gather(hidden_states.flatten(1))[:num_tokens]
        return gathered.view(-1, hc_mult, hidden_size)

    @staticmethod
    def _shard_to_context(x, forward_context):
        target = getattr(
            forward_context,
            "padded_length",
            forward_context.padded_num_tokens,
        )
        if x.shape[0] < target:
            pad_shape = list(x.shape)
            pad_shape[0] = target - x.shape[0]
            x = torch.cat((x, x.new_zeros(pad_shape)), dim=0)
        return sp_shard(x)

    @staticmethod
    def _scatter_tail_rows(tail, indices, num_tokens):
        output = tail.new_zeros((num_tokens, *tail.shape[1:]))
        if indices.numel():
            output.index_copy_(0, indices, tail)
        return output

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds=None,
        engram_lookups=None,
        engram_mask=None,
    ):
        if not get_pp_group().is_first_rank or not get_pp_group().is_last_rank:
            raise NotImplementedError("V4.1 eager milestone currently requires PP=1")
        use_sequence_parallel = getattr(self, "use_sequence_parallel", False)
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        if engram_lookups is None:
            lookups, token_mask = self.prepare_engram(input_ids, positions)
        else:
            lookups, token_mask = engram_lookups, engram_mask
        full_input_ids = input_ids
        full_lookups = lookups
        full_token_mask = token_mask
        self.shared_attention_state.reset()
        full_num_tokens = positions.shape[0]
        outer_context = get_forward_context()
        outer_flash_comm = getattr(outer_context, "flash_comm_v1_enabled", False)
        # Decoder SWA bounded replay (additional_config, default on): in an eager
        # forward (prefill / dummy) only the per-request 128-token window runs the
        # decoder layers 21-39. AscendConfig rejects this together with DSA-CP, and
        # the profile run always takes the full path so the KV cache budget stays
        # honest.
        use_decoder_tail = (
            outer_context.cudagraph_runtime_mode == CUDAGraphMode.NONE
            and not getattr(outer_context, "in_profile_run", False)
            and get_ascend_config().enable_decoder_swa_bounded_replay
        )
        if use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding,
                    hidden_states,
                )
            hidden_states = sp_shard(hidden_states)
            # The Engram buffers are zero-padded to the static token capacity,
            # so shard only the token span carried by input_ids.
            token_span = input_ids.shape[0]
            input_ids = sp_shard(input_ids)
            token_mask = sp_shard(token_mask[:token_span])
            lookups = {layer_idx: sp_shard(lookup[:token_span]) for layer_idx, lookup in lookups.items()}
        elif is_forward_context_available() and getattr(get_forward_context(), "flash_comm_v1_enabled", False):
            # FlashComm1 already reduce sequence-shards the embedding output.
            # Keep token-aligned side inputs on the same TP shard without
            # sharding hidden_states a second time.
            # The Engram buffers are zero-padded to the static token capacity,
            # so shard the token span that matches input_ids rather than the
            # whole buffer: sharding the buffer hands the later TP ranks a slice
            # of the zero padding, which silently disables the n-gram gate.
            token_span = input_ids.shape[0]
            input_ids = sp_shard(input_ids)
            token_mask = sp_shard(token_mask[:token_span])
            lookups = {layer_idx: sp_shard(lookup[:token_span]) for layer_idx, lookup in lookups.items()}
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        pre_mix = hidden_states.new_zeros(hidden_states.shape[0], self.hc_mult, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        if not use_decoder_tail:
            hidden_states, pre_mix, last_layer, aux_hidden_states = self._forward_layers(
                self.layers,
                positions,
                hidden_states,
                pre_mix,
                input_ids,
                lookups,
                token_mask,
                full_num_tokens,
            )
            assert last_layer is not None
            hidden_states = last_layer.hc_collapse(hidden_states, pre_mix)
            if use_sequence_parallel:
                hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
            hidden_states = self.norm(hidden_states)
            if aux_hidden_states:
                return hidden_states, aux_hidden_states
            return hidden_states

        early_layers = self.layers[: self.decoder_tail_start_layer]
        late_layers = self.layers[self.decoder_tail_start_layer :]
        hidden_states, pre_mix, _, aux_hidden_states = self._forward_layers(
            early_layers,
            positions,
            hidden_states,
            pre_mix,
            input_ids,
            lookups,
            token_mask,
            full_num_tokens,
        )

        if use_sequence_parallel or outer_flash_comm:
            hidden_states = self._gather_hc_rows(
                hidden_states,
                full_num_tokens,
                self.hc_mult,
                self.config.hidden_size,
            )
            pre_mix = sp_all_gather(pre_mix)[:full_num_tokens]

        attn_metadata = outer_context.attn_metadata
        first_swa = self.layers[0].self_attn.dsa_attn.swa_cache_layer
        tail_indices = torch.empty(0, dtype=torch.long, device=positions.device)
        tail_plan = None
        if attn_metadata is not None:
            representative = attn_metadata[first_swa.prefix]
            tail_plan = build_decoder_tail_plan(
                representative,
                self.config.sliding_window,
                getattr(outer_context, "num_computed_tokens_cpu", None),
                getattr(outer_context, "num_prompt_tokens_cpu", None),
            )
            tail_indices = tail_plan.token_indices_cpu.to(positions.device)

        # DP-wide decision. All ranks must agree, otherwise the nested tail
        # context's DP coordination desynchronises.
        local_tail_tokens = tail_indices.numel()
        participating = any_dp_rank_has_tail(local_tail_tokens)
        tp_size = get_tensor_model_parallel_world_size()
        work_rows = decoder_tail_work_rows(local_tail_tokens, participating, tp_size, outer_flash_comm)

        if work_rows == 0:
            # No DP rank holds rows inside its global window: skip the late
            # layers entirely (no nested forward context, no collectives).
            hidden_states = hidden_states.new_zeros((0, self.config.hidden_size))
            late_aux = [
                hidden_states for layer in late_layers if layer.layer_idx + 1 in self.aux_hidden_state_layers
            ]
        else:
            pad_rows = work_rows - local_tail_tokens
            if pad_rows:
                # This rank has nothing to do but a peer does: run one padded
                # row so the tail collectives stay aligned. Pads reuse row 0 and
                # carry slot_mapping = -1, so they write no cache. With
                # flashcomm1 the TP-aligned pad count may exceed one row.
                if tail_plan is not None and attn_metadata is not None and representative.num_reqs >= 1:
                    tail_plan = extend_decoder_tail_plan_with_pads(tail_plan, pad_rows)
                    tail_metadata = build_decoder_tail_metadata(attn_metadata, tail_plan)
                    tail_metadata = blank_padded_slots(tail_metadata, local_tail_tokens)
                    num_actual_tail_tokens = tail_metadata[first_swa.prefix].num_actual_tokens
                else:
                    tail_metadata = None
                    num_actual_tail_tokens = work_rows
                work_indices = tail_plan.token_indices_cpu.to(positions.device) if tail_plan is not None else None
                if work_indices is None:
                    work_indices = torch.zeros(work_rows, dtype=torch.long, device=positions.device)
            else:
                work_indices = tail_indices
                tail_metadata = build_decoder_tail_metadata(attn_metadata, tail_plan)
                num_actual_tail_tokens = tail_metadata[first_swa.prefix].num_actual_tokens

            hidden_states = hidden_states.index_select(0, work_indices)
            pre_mix = pre_mix.index_select(0, work_indices)
            tail_positions = positions.index_select(0, work_indices)
            tail_input_ids = full_input_ids.index_select(0, work_indices)
            tail_token_mask = (
                full_token_mask.index_select(0, work_indices) if full_token_mask.numel() else full_token_mask
            )
            tail_lookups = {
                layer_idx: lookup.index_select(0, work_indices) for layer_idx, lookup in full_lookups.items()
            }

            topk = self.shared_attention_state.topk_indices.index_select(0, work_indices).clone()
            candidates = self.shared_attention_state.candidates.index_select(0, work_indices).clone()
            self.shared_attention_state.topk_indices[:work_rows].copy_(topk)
            self.shared_attention_state.candidates[:work_rows].copy_(candidates)
            tail_num_tokens = work_rows

        if work_rows:
            from vllm_ascend.ascend_forward_context import set_ascend_forward_context

            with set_ascend_forward_context(
                tail_metadata,
                self.vllm_config,
                num_tokens=tail_num_tokens,
                in_profile_run=getattr(outer_context, "in_profile_run", False),
                num_actual_tokens=num_actual_tail_tokens,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                model_instance=getattr(outer_context, "model_instance", None),
                skip_compiled=True,
                has_sinks=getattr(outer_context, "sinks", False),
                input_ids=tail_input_ids,
                eplb_heat_collection_status=getattr(
                    outer_context,
                    "eplb_heat_collection_status",
                    False,
                ),
            ):
                tail_context = get_forward_context()
                tail_flash_comm = getattr(tail_context, "flash_comm_v1_enabled", False)
                has_tail_tokens = bool(tail_context.max_tokens_across_dp)
                # Defensive only: every rank enters this branch with work_rows >= 1
                # (participating is the DP-wide OR, so it is true on all ranks), hence
                # the nested context always reports at least one token. The branch
                # keeps the empty-output shape if that invariant ever breaks.
                if not has_tail_tokens:
                    hidden_states = hidden_states.new_zeros((0, self.config.hidden_size))
                    late_aux = [
                        hidden_states for layer in late_layers if layer.layer_idx + 1 in self.aux_hidden_state_layers
                    ]
                elif use_sequence_parallel:
                    hidden_states = sp_shard(hidden_states)
                    pre_mix = sp_shard(pre_mix)
                    tail_input_ids = sp_shard(tail_input_ids)
                    tail_token_mask = sp_shard(tail_token_mask)
                    tail_lookups = {layer_idx: sp_shard(lookup) for layer_idx, lookup in tail_lookups.items()}
                elif tail_flash_comm:
                    hidden_states = self._shard_to_context(hidden_states, tail_context)
                    pre_mix = self._shard_to_context(pre_mix, tail_context)
                    _tp_ids = get_tp_group().world_size
                    _ids_target = hidden_states.shape[0] * _tp_ids
                    if tail_input_ids.shape[0] < _ids_target:
                        _ps = list(tail_input_ids.shape)
                        _ps[0] = _ids_target - tail_input_ids.shape[0]
                        tail_input_ids = torch.cat(
                            (tail_input_ids, tail_input_ids.new_zeros(_ps)), dim=0
                        )
                    # MoE reads forward_context.input_ids; publish the padded copy.
                    tail_context.input_ids = tail_input_ids
                    if tail_token_mask.numel():
                        tail_token_mask = self._shard_to_context(
                            tail_token_mask,
                            tail_context,
                        )
                    tail_lookups = {
                        layer_idx: self._shard_to_context(lookup, tail_context)
                        for layer_idx, lookup in tail_lookups.items()
                    }

                if has_tail_tokens:
                    hidden_states, pre_mix, last_layer, late_aux = self._forward_layers(
                        late_layers,
                        tail_positions,
                        hidden_states,
                        pre_mix,
                        tail_input_ids,
                        tail_lookups,
                        tail_token_mask,
                        tail_num_tokens,
                    )
                    assert last_layer is not None
                    hidden_states = last_layer.hc_collapse(hidden_states, pre_mix)
                    if tail_flash_comm or use_sequence_parallel:
                        hidden_states = sp_all_gather(hidden_states)[:tail_num_tokens]
                    if tail_flash_comm and not use_sequence_parallel:
                        late_aux = [sp_all_gather(aux_hidden_state)[:tail_num_tokens] for aux_hidden_state in late_aux]
                    hidden_states = self.norm(hidden_states)

        # The late layers ran on the padded tail batch; write back only this rank's
        # real tail rows (the throwaway rows sit at the end).
        if 0 < local_tail_tokens < tail_num_tokens:
            hidden_states = hidden_states[:local_tail_tokens]
            late_aux = [aux[:local_tail_tokens] for aux in late_aux]
        hidden_states = self._scatter_tail_rows(
            hidden_states,
            tail_indices,
            full_num_tokens,
        )
        late_aux = [
            self._scatter_tail_rows(aux_hidden_state, tail_indices, full_num_tokens) for aux_hidden_state in late_aux
        ]
        if outer_flash_comm:
            hidden_states = self._shard_to_context(hidden_states, outer_context)
            late_aux = [self._shard_to_context(aux_hidden_state, outer_context) for aux_hidden_state in late_aux]
        aux_hidden_states.extend(late_aux)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class AscendDeepseekV41ForCausalLM(AscendDeepseekV4ForCausalLM):
    model_cls = DeepseekV41Model
    requires_raw_input_tokens = True
    # The vision-only router bias (`gate.bias_vl`) is instantiated by
    # DeepseekV4MoE whenever the checkpoint declares a vision tower, so it must
    # be loaded rather than deferred. Only the vision tower / aligner / image
    # sentinel parameters remain deferred to the multimodal wrapper.
    _DEFERRED_WEIGHT_MARKERS = ()
    _DEFERRED_WEIGHT_PREFIXES = ("aligner.", "vision.", "image_", "mtp.")

    def prepare_engram_inputs(self, input_ids, positions, padded_tokens=None):
        return self.model.prepare_engram_inputs(input_ids, positions, padded_tokens)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        engram_lookups=None,
        engram_mask=None,
    ):
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            engram_lookups=engram_lookups,
            engram_mask=engram_mask,
        )

    @classmethod
    def _is_milestone_weight(cls, name):
        return not name.startswith(cls._DEFERRED_WEIGHT_PREFIXES) and not any(
            marker in name for marker in cls._DEFERRED_WEIGHT_MARKERS
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if not get_ascend_config().enable_engram:
            return super().load_weights((name, tensor) for name, tensor in weights if ".engram." not in name)
        engram_loaded = set()

        def milestone_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                if ".engram." in name:
                    # Bypass V4's generic embed -> embed_tokens remapping and TP loader.
                    local_name = name.removeprefix("model.")
                    # Compressed Engram scales are consumed by the shard loader.
                    if local_name.endswith(".engram.embed.scale"):
                        continue
                    parameter_name = "model." + local_name
                    if local_name.endswith(".engram.embed.weight"):
                        layer_id = int(local_name.split(".")[1])
                        self.model.layers[layer_id].engram.embed.load_checkpoint(
                            self.model.engram_weight_root, local_name
                        )
                    else:
                        param = self.get_parameter(parameter_name)
                        if tensor.dtype != torch.bfloat16 or tensor.shape != param.shape:
                            raise ValueError(f"Unexpected BF16 Engram parameter: {name}")
                        param.data.copy_(tensor)
                    engram_loaded.add(parameter_name)
                elif self._is_milestone_weight(name):
                    yield name, tensor

        loaded = super().load_weights(milestone_weights())
        expected = {name for name, _ in self.named_parameters() if ".engram." in name}
        if engram_loaded != expected:
            raise ValueError(f"Missing Engram weights: {expected - engram_loaded}")
        return loaded | engram_loaded
