# SPDX-License-Identifier: Apache-2.0
"""Bit-exact query layouts, retained split widths, and changing graph inputs."""

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.ops.triton.dcp.sfa_dcp_query import prepare_query_head_major, unpack_query  # noqa: E402


@pytest.mark.parametrize("widths", [(512, 64)])
@pytest.mark.parametrize("tokens,heads", [(2, 3), (6, 8), (12, 1)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("column_stride", [1, 2])
def test_query_bytes_and_changing_graph_inputs(widths, tokens, heads, dtype, column_stride):
    torch.npu.set_device(0)
    owners = [torch.empty(heads, tokens, width * column_stride, dtype=dtype, device="npu") for width in widths]
    qn, qr = [owner.transpose(0, 1)[..., ::column_stride] for owner in owners]

    def invoke():
        packed = prepare_query_head_major(qn, qr)
        return packed, unpack_query(packed)

    for _ in range(3):
        invoke()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        packed, fragments = invoke()
    for step in range(3):
        for owner in owners:
            bits = torch.arange(owner.numel()).mul(17).add(step * 4099).to(torch.int16).reshape(owner.shape)
            owner.copy_(bits.view(dtype))
        graph.replay()
        expected = torch.cat((qn.cpu(), qr.cpu()), dim=-1).transpose(0, 1).contiguous()
        assert torch.equal(packed.cpu().view(torch.int16), expected.view(torch.int16))
        for actual, source in zip(fragments, (qn, qr)):
            assert actual.is_contiguous()
            assert torch.equal(actual.cpu().view(torch.int16), source.cpu().view(torch.int16))
    torch.npu.synchronize()
