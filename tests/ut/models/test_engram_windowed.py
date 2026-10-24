# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Late WKV scheduling, SP row ownership and exceptional retirement."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.models.deepseek_v41")

from vllm_ascend.models.deepseek_v41 import model as model_mod
from vllm_ascend.models.deepseek_v41.engram.layer import AscendEngram
from vllm_ascend.models.deepseek_v41.engram.windowed import WindowedEngramProjection


@pytest.fixture
def window_runtime(monkeypatch):
    calls = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            calls.append(("wait", self.name, event.name))

        def wait_stream(self, stream):
            calls.append(("join", self.name, stream.name))

    class Event:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            calls.append(("record", stream.name, self.name))

        def wait(self, stream):
            calls.append(("external_wait", stream.name, self.name))

        def reset(self, stream):
            calls.append(("reset", stream.name, self.name))

    main, side = Stream("main"), Stream("side")
    current = [main]

    @contextmanager
    def on_stream(stream):
        old = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = old

    monkeypatch.setattr(torch.npu, "current_stream", lambda: current[0])
    monkeypatch.setattr(torch.npu, "stream", on_stream)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: calls.append(("retain", stream.name)))
    return calls, side, Event


def make_projection(runtime, failure=None):
    calls, side, event = runtime
    rows = torch.arange(24, dtype=torch.bfloat16).view(6, 4) / 8
    weight = torch.arange(24, dtype=torch.bfloat16).view(6, 4) / 16
    output = torch.full((5, 6), float("nan"), dtype=torch.bfloat16)

    def shard(values):
        calls.append(("shard",))
        return values[1::2]

    def project(values, out, begin, end):
        calls.append(("project", begin, end))
        if begin // 2 == failure:
            raise RuntimeError("projection failed")
        torch.mm(values, weight[begin:end].t(), out=out)

    projection = WindowedEngramProjection(
        SimpleNamespace(dim=2, project_chunk_into=project),
        rows,
        output,
        side,
        [event(str(i)) for i in range(6)],
        first_layer=1,
        consumer_layer=7,
        tokens=3,
        shard=shard,
        lookup_ready=event("rows"),
        external=True,
    )
    return projection, torch.nn.functional.linear(rows[1::2], weight)


def test_windowed_projection_waits_once_and_uses_only_sp_rows(window_runtime):
    projection, expected = make_projection(window_runtime)
    calls = window_runtime[0]
    with projection:
        for index in range(6):
            projection.submit_window(index)
        actual = projection.finish()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isnan(projection.output[3:]).all()
    assert calls.count(("shard",)) == 1
    assert calls.count(("external_wait", "side", "rows")) == 1
    assert calls.count(("reset", "side", "rows")) == 1
    assert calls.index(("reset", "side", "rows")) < calls.index(("shard",))
    assert calls.index(("shard",)) < calls.index(("project", 0, 2))
    assert calls.count(("join", "main", "side")) == 1


def test_windowed_projection_finishes_pieces_after_skipped_windows(window_runtime):
    projection, expected = make_projection(window_runtime)
    with projection:
        projection.submit_window(0)
        actual = projection.finish()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert projection.next_chunk == 3


def test_no_window_keeps_original_lookup_event_unconsumed(window_runtime):
    projection, _ = make_projection(window_runtime)
    with projection:
        projection.submit_window(1)
    assert not projection.started
    assert not window_runtime[0]


@pytest.mark.parametrize("failure", [0, 1])
def test_projection_error_joins_even_before_first_piece_completes(window_runtime, failure):
    projection, _ = make_projection(window_runtime, failure)
    with pytest.raises(RuntimeError, match="projection failed"), projection:
        projection.submit_window(0)
        projection.submit_window(2)
    assert window_runtime[0][-1] == ("join", "main", "side")


def test_decoder_opens_window_before_sp_reduce_scatter(monkeypatch):
    calls = []
    layer = SimpleNamespace(use_sequence_parallel=True)
    for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"):
        setattr(layer, name, None)
    layer.hc_pre = lambda x, *args: (x.mean(1), None, None, None)
    layer.input_layernorm = lambda x: x
    layer.self_attn = lambda *args: calls.append("attention") or args[1]
    layer.hc_post = lambda x, residual, *args: calls.append("hc_post") or residual
    layer.rms_norm_cast = lambda x: (x, x.float())
    layer.mlp = lambda x, **kwargs: x
    monkeypatch.setattr(model_mod, "sp_all_gather", lambda x: x)
    monkeypatch.setattr(model_mod, "sp_reduce_scatter", lambda x: calls.append("reduce_scatter") or x)
    hidden = torch.ones(3, 4, 8)
    result, _ = model_mod.DeepseekV41DecoderLayer.forward(
        layer, torch.arange(3), hidden, None, engram_wkv_window=lambda: calls.append("window")
    )
    assert calls[:4] == ["attention", "window", "reduce_scatter", "hc_post"]
    torch.testing.assert_close(result, hidden)


def test_quantized_projection_is_not_sliced():
    layer = object.__new__(AscendEngram)
    torch.nn.Module.__init__(layer)
    layer.dim, layer.hc_mult = 2, 2
    layer.wkv = torch.nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
    assert layer.supports_windowed_projection()
    layer.wkv.quant_method = SimpleNamespace()
    assert not layer.supports_windowed_projection()
