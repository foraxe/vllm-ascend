# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""One-NPU oracle for the DSA-CP C128 local-compressor prefix state.

The production TP8 path replaces one 5120-token compressor invocation with
one 640-token invocation on each rank.  Every rank retains its own replicated
compressor state, then the non-aligned 3080-token tail falls back to the
global compressor path.  This test reproduces those eight rank-local state
histories on one NPU and compares them with the unsharded reference.

Run on an A3 target image:

.. code-block:: bash

   ASCEND_RT_VISIBLE_DEVICES=0 \
   DSA_CP_ORACLE_RESULT_JSON=/path/to/result.json \
   pytest -sv tests/e2e/attention/test_dsa_cp_c128_prefix_state_npu.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest
import torch

_SEED = 20260730
_TP_SIZE = 8
_COMPRESS_RATIO = 128
_PREFIX_TOKENS = 5120
_LOCAL_TOKENS = _PREFIX_TOKENS // _TP_SIZE
_TAIL_TOKENS = 3080
_TOTAL_TOKENS = _PREFIX_TOKENS + _TAIL_TOKENS
_CONTINUATION_TOKENS = 24
_HIDDEN_SIZE = 7168
_COMPRESSED_SIZE = 512
_ROPE_HEAD_DIM = 64
_STATE_BLOCK_SIZE = 32
_STATE_BLOCKS = 258
_STATE_BLOCK_TABLE_ENTRIES = 257
_STATE_PADDED_DIM = 1024
_LIVE_STATE_START = _TOTAL_TOKENS - _COMPRESS_RATIO
_OUTPUT_ATOL = 5e-2
_OUTPUT_RTOL = 1e-2
_STATE_ATOL = 1e-4
_STATE_RTOL = 1e-3


def _make_rope(start_pos: int, token_count: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Match production completed-group positions plus zero-position padding."""
    valid_rows = _valid_output_rows(start_pos, token_count)
    row_count = min(
        token_count,
        token_count // _COMPRESS_RATIO + 1,
    )
    padding_rows = row_count - valid_rows
    if padding_rows < 0:
        raise ValueError(
            "compressor output rows cannot represent completed C128 groups: "
            f"start_pos={start_pos}, token_count={token_count}"
        )
    first_group_start = start_pos - start_pos % _COMPRESS_RATIO
    valid_positions = (
        first_group_start
        + torch.arange(valid_rows, dtype=torch.float32)
        * _COMPRESS_RATIO
    )
    # ``_get_padded_compressed_position`` pads missing per-request rows with
    # input position zero. ``slice_c128_local_compressor_rope`` then carries
    # this same final row into every aligned local invocation.
    row_positions = torch.cat(
        (
            valid_positions,
            torch.zeros(padding_rows, dtype=torch.float32),
        )
    )
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(_ROPE_HEAD_DIM, dtype=torch.float32)
        / _ROPE_HEAD_DIM
    )
    angles = row_positions[:, None] * frequencies[None, :]
    return angles.sin().to(device=device), angles.cos().to(device=device)


def _valid_output_rows(start_pos: int, token_count: int) -> int:
    """Count completed C128 groups; the custom op's final row is padding."""
    return (start_pos % _COMPRESS_RATIO + token_count) // _COMPRESS_RATIO


def _metric(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """Return finite CPU-side comparison metrics without hiding large errors."""
    actual_float = actual.float().cpu()
    expected_float = expected.float().cpu()
    absolute = (actual_float - expected_float).abs()
    relative = absolute / expected_float.abs().clamp_min(1e-6)
    has_elements = absolute.numel() > 0
    tolerance = _OUTPUT_ATOL + _OUTPUT_RTOL * expected_float.abs()
    if absolute.ndim > 1:
        failing_rows = (
            torch.logical_or(~torch.isfinite(absolute), absolute > tolerance)
            .flatten(1)
            .any(dim=1)
            .nonzero()
            .flatten()
            .tolist()
        )
        row_max_abs = torch.nan_to_num(
            absolute.flatten(1),
            nan=math.inf,
            posinf=math.inf,
            neginf=math.inf,
        ).max(dim=1).values.tolist()
    else:
        failing_rows = []
        row_max_abs = []
    return {
        "shape": list(actual.shape),
        "max_abs": float(absolute.max().item()) if has_elements else 0.0,
        "max_rel": float(relative.max().item()) if has_elements else 0.0,
        "actual_sha256": hashlib.sha256(
            actual_float.contiguous().numpy().tobytes()
        ).hexdigest(),
        "expected_sha256": hashlib.sha256(
            expected_float.contiguous().numpy().tobytes()
        ).hexdigest(),
        "equal_hash": bool(torch.equal(actual_float, expected_float)),
        "actual_nonfinite": int((~torch.isfinite(actual_float)).sum().item()),
        "expected_nonfinite": int((~torch.isfinite(expected_float)).sum().item()),
        "failing_rows_at_output_tolerance": failing_rows,
        "row_max_abs": row_max_abs,
    }


def test_empty_metric_is_well_defined() -> None:
    """A continuation with no completed C128 rows has a valid empty metric."""
    empty = torch.empty((0, _COMPRESSED_SIZE), dtype=torch.bfloat16)

    metric = _metric(empty, empty.clone())

    assert metric["shape"] == [0, _COMPRESSED_SIZE]
    assert metric["max_abs"] == 0.0
    assert metric["max_rel"] == 0.0
    assert metric["actual_nonfinite"] == 0
    assert metric["expected_nonfinite"] == 0
    assert metric["failing_rows_at_output_tolerance"] == []
    assert metric["row_max_abs"] == []
    assert metric["equal_hash"]


def test_rope_uses_production_zero_position_padding() -> None:
    """The final compressor ABI row must encode input position zero."""
    sin, cos = _make_rope(
        _PREFIX_TOKENS,
        _TAIL_TOKENS,
        device=torch.device("cpu"),
    )

    assert sin.shape == (25, _ROPE_HEAD_DIM)
    assert cos.shape == (25, _ROPE_HEAD_DIM)
    torch.testing.assert_close(sin[-1], torch.zeros_like(sin[-1]))
    torch.testing.assert_close(cos[-1], torch.ones_like(cos[-1]))


def _logical_state_slice(
    state_cache: torch.Tensor,
    state_block_table: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    logical_blocks = state_cache.index_select(
        0,
        state_block_table[0].to(dtype=torch.int64),
    )
    return logical_blocks.flatten(0, 1)[start:end].clone()


def _run_compressor(
    hidden_states: torch.Tensor,
    *,
    start_pos: int,
    state_cache: torch.Tensor,
    state_block_table: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    ape: torch.Tensor,
    norm: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> torch.Tensor:
    token_count = hidden_states.shape[0]
    cu_seqlens = torch.tensor(
        [0, token_count],
        dtype=torch.int32,
        device=hidden_states.device,
    )
    start_positions = torch.tensor(
        [start_pos],
        dtype=torch.int32,
        device=hidden_states.device,
    )
    return torch.ops._C_ascend.compressor(
        hidden_states,
        wkv,
        wgate,
        state_cache,
        ape,
        norm,
        sin,
        cos,
        state_block_table=state_block_table,
        cu_seqlens=cu_seqlens,
        seqused=None,
        start_pos=start_positions,
        rope_head_dim=_ROPE_HEAD_DIM,
        cmp_ratio=_COMPRESS_RATIO,
        coff=1,
        norm_eps=1e-6,
        rotary_mode=2,
        cache_mode=1,
    )


def test_c128_local_prefix_preserves_tail_and_continuation_state() -> None:
    """All eight local prefix histories must match the global state history."""
    torch_npu = pytest.importorskip("torch_npu")
    from vllm_ascend.attention.context_parallel.c128_owner_cache import (
        C128LocalCompressorPlan,
        slice_c128_local_compressor_output,
        slice_c128_local_compressor_rope,
    )
    from vllm_ascend.utils import enable_custom_op

    torch_npu.npu.set_device(0)
    assert enable_custom_op(), "vLLM Ascend custom operators are unavailable"
    device = torch.device("npu:0")
    torch.manual_seed(_SEED)
    torch_npu.npu.manual_seed_all(_SEED)

    hidden_states = torch.randn(
        (_TOTAL_TOKENS + _CONTINUATION_TOKENS, _HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    ).mul_(0.02)
    wkv = torch.randn(
        (_COMPRESSED_SIZE, _HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    ).mul_(0.02)
    wgate = torch.randn(
        (_COMPRESSED_SIZE, _HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    ).mul_(0.02)
    ape = torch.randn(
        (_COMPRESS_RATIO, _COMPRESSED_SIZE),
        dtype=torch.float32,
        device=device,
    ).mul_(0.02)
    norm = torch.ones((_COMPRESSED_SIZE,), dtype=torch.bfloat16, device=device)
    initial_state = torch.zeros(
        (
            _STATE_BLOCKS,
            _STATE_BLOCK_SIZE,
            _STATE_PADDED_DIM,
        ),
        dtype=torch.float32,
        device=device,
    )
    state_block_table = torch.arange(
        1,
        _STATE_BLOCK_TABLE_ENTRIES + 1,
        dtype=torch.int32,
        device=device,
    ).view(1, -1)

    prefix_sin, prefix_cos = _make_rope(0, _PREFIX_TOKENS, device=device)
    tail_sin, tail_cos = _make_rope(
        _PREFIX_TOKENS,
        _TAIL_TOKENS,
        device=device,
    )
    continuation_sin, continuation_cos = _make_rope(
        _TOTAL_TOKENS,
        _CONTINUATION_TOKENS,
        device=device,
    )

    reference_state = initial_state.clone()
    reference_prefix = _run_compressor(
        hidden_states[:_PREFIX_TOKENS],
        start_pos=0,
        state_cache=reference_state,
        state_block_table=state_block_table,
        wkv=wkv,
        wgate=wgate,
        ape=ape,
        norm=norm,
        sin=prefix_sin,
        cos=prefix_cos,
    )
    reference_prefix = reference_prefix[: _PREFIX_TOKENS // _COMPRESS_RATIO].clone()
    reference_tail_raw = _run_compressor(
        hidden_states[_PREFIX_TOKENS:_TOTAL_TOKENS],
        start_pos=_PREFIX_TOKENS,
        state_cache=reference_state,
        state_block_table=state_block_table,
        wkv=wkv,
        wgate=wgate,
        ape=ape,
        norm=norm,
        sin=tail_sin,
        cos=tail_cos,
    ).clone()
    tail_valid_rows = _valid_output_rows(
        _PREFIX_TOKENS,
        _TAIL_TOKENS,
    )
    reference_tail = reference_tail_raw[:tail_valid_rows]
    reference_tail_padding = reference_tail_raw[tail_valid_rows:]
    reference_live_state = _logical_state_slice(
        reference_state,
        state_block_table,
        _LIVE_STATE_START,
        _TOTAL_TOKENS,
    )
    reference_continuation_raw = _run_compressor(
        hidden_states[_TOTAL_TOKENS:],
        start_pos=_TOTAL_TOKENS,
        state_cache=reference_state,
        state_block_table=state_block_table,
        wkv=wkv,
        wgate=wgate,
        ape=ape,
        norm=norm,
        sin=continuation_sin,
        cos=continuation_cos,
    ).clone()
    continuation_valid_rows = _valid_output_rows(
        _TOTAL_TOKENS,
        _CONTINUATION_TOKENS,
    )
    reference_continuation = reference_continuation_raw[
        :continuation_valid_rows
    ]
    reference_continuation_padding = reference_continuation_raw[
        continuation_valid_rows:
    ]
    reference_future_state = _logical_state_slice(
        reference_state,
        state_block_table,
        _TOTAL_TOKENS + _CONTINUATION_TOKENS - _COMPRESS_RATIO,
        _TOTAL_TOKENS + _CONTINUATION_TOKENS,
    )
    torch_npu.npu.synchronize()

    control_state = initial_state.clone()
    _run_compressor(
        hidden_states[:_PREFIX_TOKENS],
        start_pos=0,
        state_cache=control_state,
        state_block_table=state_block_table,
        wkv=wkv,
        wgate=wgate,
        ape=ape,
        norm=norm,
        sin=prefix_sin,
        cos=prefix_cos,
    )
    control_tail_raw = _run_compressor(
        hidden_states[_PREFIX_TOKENS:_TOTAL_TOKENS],
        start_pos=_PREFIX_TOKENS,
        state_cache=control_state,
        state_block_table=state_block_table,
        wkv=wkv,
        wgate=wgate,
        ape=ape,
        norm=norm,
        sin=tail_sin,
        cos=tail_cos,
    ).clone()
    control_tail = control_tail_raw[:tail_valid_rows]
    control_tail_padding = control_tail_raw[tail_valid_rows:]
    control_live_state = _logical_state_slice(
        control_state,
        state_block_table,
        _LIVE_STATE_START,
        _TOTAL_TOKENS,
    )
    control_continuation_raw = _run_compressor(
        hidden_states[_TOTAL_TOKENS:],
        start_pos=_TOTAL_TOKENS,
        state_cache=control_state,
        state_block_table=state_block_table,
        wkv=wkv,
        wgate=wgate,
        ape=ape,
        norm=norm,
        sin=continuation_sin,
        cos=continuation_cos,
    ).clone()
    control_continuation = control_continuation_raw[
        :continuation_valid_rows
    ]
    control_continuation_padding = control_continuation_raw[
        continuation_valid_rows:
    ]
    control_future_state = _logical_state_slice(
        control_state,
        state_block_table,
        _TOTAL_TOKENS + _CONTINUATION_TOKENS - _COMPRESS_RATIO,
        _TOTAL_TOKENS + _CONTINUATION_TOKENS,
    )
    torch_npu.npu.synchronize()
    control_checks = {
        "tail_output": bool(
            torch.allclose(
                control_tail,
                reference_tail,
                atol=_OUTPUT_ATOL,
                rtol=_OUTPUT_RTOL,
            )
        ),
        "live_state_8072_8200": bool(
            torch.allclose(
                control_live_state,
                reference_live_state,
                atol=_STATE_ATOL,
                rtol=_STATE_RTOL,
            )
        ),
        "continuation_output": bool(
            torch.allclose(
                control_continuation,
                reference_continuation,
                atol=_OUTPUT_ATOL,
                rtol=_OUTPUT_RTOL,
            )
        ),
        "future_live_state": bool(
            torch.allclose(
                control_future_state,
                reference_future_state,
                atol=_STATE_ATOL,
                rtol=_STATE_RTOL,
            )
        ),
        "tail_padding_raw_equal": bool(
            torch.equal(
                control_tail_padding,
                reference_tail_padding,
            )
        ),
        "continuation_padding_raw_equal": bool(
            torch.equal(
                control_continuation_padding,
                reference_continuation_padding,
            )
        ),
    }
    control_metrics = {
        "tail_output": _metric(control_tail, reference_tail),
        "live_state_8072_8200": _metric(
            control_live_state,
            reference_live_state,
        ),
        "continuation_output": _metric(
            control_continuation,
            reference_continuation,
        ),
        "future_live_state": _metric(
            control_future_state,
            reference_future_state,
        ),
        "tail_padding_diagnostic": _metric(
            control_tail_padding,
            reference_tail_padding,
        ),
        "continuation_padding_diagnostic": _metric(
            control_continuation_padding,
            reference_continuation_padding,
        ),
    }
    control_passed = all(
        control_checks[name]
        for name in (
            "tail_output",
            "live_state_8072_8200",
            "continuation_output",
            "future_live_state",
        )
    )

    rank_results: list[dict[str, Any]] = []
    all_passed = control_passed
    for rank in range(_TP_SIZE):
        slot_start = rank * (_LOCAL_TOKENS // _COMPRESS_RATIO)
        slot_end = slot_start + (_LOCAL_TOKENS // _COMPRESS_RATIO)
        plan = C128LocalCompressorPlan(
            slot_start=slot_start,
            slot_end=slot_end,
        )
        local_start = rank * _LOCAL_TOKENS
        local_end = local_start + _LOCAL_TOKENS
        local_sin = slice_c128_local_compressor_rope(prefix_sin, plan)
        local_cos = slice_c128_local_compressor_rope(prefix_cos, plan)

        candidate_state = initial_state.clone()
        candidate_prefix = _run_compressor(
            hidden_states[local_start:local_end],
            start_pos=local_start,
            state_cache=candidate_state,
            state_block_table=state_block_table,
            wkv=wkv,
            wgate=wgate,
            ape=ape,
            norm=norm,
            sin=local_sin,
            cos=local_cos,
        )
        candidate_prefix = slice_c128_local_compressor_output(
            candidate_prefix,
            plan,
        ).clone()
        candidate_tail_raw = _run_compressor(
            hidden_states[_PREFIX_TOKENS:_TOTAL_TOKENS],
            start_pos=_PREFIX_TOKENS,
            state_cache=candidate_state,
            state_block_table=state_block_table,
            wkv=wkv,
            wgate=wgate,
            ape=ape,
            norm=norm,
            sin=tail_sin,
            cos=tail_cos,
        ).clone()
        candidate_tail = candidate_tail_raw[:tail_valid_rows]
        candidate_tail_padding = candidate_tail_raw[tail_valid_rows:]
        candidate_live_state = _logical_state_slice(
            candidate_state,
            state_block_table,
            _LIVE_STATE_START,
            _TOTAL_TOKENS,
        )
        candidate_continuation_raw = _run_compressor(
            hidden_states[_TOTAL_TOKENS:],
            start_pos=_TOTAL_TOKENS,
            state_cache=candidate_state,
            state_block_table=state_block_table,
            wkv=wkv,
            wgate=wgate,
            ape=ape,
            norm=norm,
            sin=continuation_sin,
            cos=continuation_cos,
        ).clone()
        candidate_continuation = candidate_continuation_raw[
            :continuation_valid_rows
        ]
        candidate_continuation_padding = candidate_continuation_raw[
            continuation_valid_rows:
        ]
        candidate_future_state = _logical_state_slice(
            candidate_state,
            state_block_table,
            _TOTAL_TOKENS + _CONTINUATION_TOKENS - _COMPRESS_RATIO,
            _TOTAL_TOKENS + _CONTINUATION_TOKENS,
        )
        torch_npu.npu.synchronize()

        expected_prefix = reference_prefix[slot_start:slot_end]
        metrics = {
            "prefix_output": _metric(candidate_prefix, expected_prefix),
            "tail_output": _metric(candidate_tail, reference_tail),
            "live_state_8072_8200": _metric(
                candidate_live_state,
                reference_live_state,
            ),
            "continuation_output": _metric(
                candidate_continuation,
                reference_continuation,
            ),
                "future_live_state": _metric(
                    candidate_future_state,
                    reference_future_state,
                ),
                "tail_padding_diagnostic": _metric(
                    candidate_tail_padding,
                    reference_tail_padding,
                ),
                "continuation_padding_diagnostic": _metric(
                    candidate_continuation_padding,
                    reference_continuation_padding,
                ),
            }
        prefix_pass = torch.allclose(
            candidate_prefix,
            expected_prefix,
            atol=_OUTPUT_ATOL,
            rtol=_OUTPUT_RTOL,
        )
        tail_pass = torch.allclose(
            candidate_tail,
            reference_tail,
            atol=_OUTPUT_ATOL,
            rtol=_OUTPUT_RTOL,
        )
        live_state_pass = torch.allclose(
            candidate_live_state,
            reference_live_state,
            atol=_STATE_ATOL,
            rtol=_STATE_RTOL,
        )
        continuation_pass = torch.allclose(
            candidate_continuation,
            reference_continuation,
            atol=_OUTPUT_ATOL,
            rtol=_OUTPUT_RTOL,
        )
        future_state_pass = torch.allclose(
            candidate_future_state,
            reference_future_state,
            atol=_STATE_ATOL,
            rtol=_STATE_RTOL,
        )
        rank_passed = bool(
            prefix_pass
            and tail_pass
            and live_state_pass
            and continuation_pass
            and future_state_pass
        )
        rank_results.append(
            {
                "rank": rank,
                "local_start": local_start,
                "local_end": local_end,
                "slot_start": slot_start,
                "slot_end": slot_end,
                "passed": rank_passed,
                "checks": {
                    "prefix_output": bool(prefix_pass),
                    "tail_output": bool(tail_pass),
                    "live_state_8072_8200": bool(live_state_pass),
                    "continuation_output": bool(continuation_pass),
                    "future_live_state": bool(future_state_pass),
                    "tail_padding_raw_equal": bool(
                        torch.equal(
                            candidate_tail_padding,
                            reference_tail_padding,
                        )
                    ),
                    "continuation_padding_raw_equal": bool(
                        torch.equal(
                            candidate_continuation_padding,
                            reference_continuation_padding,
                        )
                    ),
                },
                "metrics": metrics,
            }
        )
        all_passed = all_passed and rank_passed

    result = {
        "status": "PASS" if all_passed else "FAIL",
        "hypothesis": (
            "Each TP8-local 640-token C128 compressor invocation followed by "
            "the shared 3080-token tail preserves the global compressor "
            "outputs, live state, and future continuation."
        ),
        "configuration": {
            "seed": _SEED,
            "device": str(device),
            "tp_size": _TP_SIZE,
            "prefix_tokens": _PREFIX_TOKENS,
            "local_tokens": _LOCAL_TOKENS,
            "tail_tokens": _TAIL_TOKENS,
            "continuation_tokens": _CONTINUATION_TOKENS,
            "tail_valid_rows": tail_valid_rows,
            "tail_padding_rows": int(
                reference_tail_padding.shape[0]
            ),
            "continuation_valid_rows": continuation_valid_rows,
            "continuation_padding_rows": int(
                reference_continuation_padding.shape[0]
            ),
            "hidden_shape": list(hidden_states.shape),
            "wkv_shape": list(wkv.shape),
            "wgate_shape": list(wgate.shape),
            "ape_shape": list(ape.shape),
            "norm_shape": list(norm.shape),
            "state_shape": list(initial_state.shape),
            "state_block_table_shape": list(state_block_table.shape),
            "state_block_table_first_physical_id": 1,
            "state_block_table_last_physical_id": (
                _STATE_BLOCK_TABLE_ENTRIES
            ),
            "prefix_rope_shape": list(prefix_sin.shape),
            "local_rope_shape": [
                _LOCAL_TOKENS // _COMPRESS_RATIO + 1,
                _ROPE_HEAD_DIM,
            ],
            "tail_rope_shape": list(tail_sin.shape),
            "rope_padding_input_position": 0,
        },
        "thresholds": {
            "output_atol": _OUTPUT_ATOL,
            "output_rtol": _OUTPUT_RTOL,
            "state_atol": _STATE_ATOL,
            "state_rtol": _STATE_RTOL,
        },
        "operator_schema": str(torch.ops._C_ascend.compressor.default._schema),
        "padding_semantics": (
            "The custom op emits one final per-batch padding row. Padding "
            "bytes are diagnostic only and are excluded from equivalence."
        ),
        "repeatability_control": {
            "passed": control_passed,
            "checks": control_checks,
            "metrics": control_metrics,
        },
        "rank_results": rank_results,
    }
    result_path_value = os.getenv("DSA_CP_ORACLE_RESULT_JSON")
    if result_path_value:
        result_path = Path(result_path_value)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    failure_summary = {
        "repeatability_control": control_checks,
        "ranks": [
            {
                "rank": rank_result["rank"],
                "checks": rank_result["checks"],
                "tail_failing_rows": rank_result["metrics"]["tail_output"][
                    "failing_rows_at_output_tolerance"
                ],
                "tail_padding_raw_equal": rank_result["checks"][
                    "tail_padding_raw_equal"
                ],
                "continuation_padding_raw_equal": rank_result["checks"][
                    "continuation_padding_raw_equal"
                ],
            }
            for rank_result in rank_results
            if not rank_result["passed"]
        ],
    }
    assert all_passed, json.dumps(failure_summary, indent=2, sort_keys=True)
