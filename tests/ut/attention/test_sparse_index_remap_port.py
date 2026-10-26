# SPDX-License-Identifier: Apache-2.0
"""Empty input must return before dividing by TopK or launching kernels."""

import ast
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("shape", [(0, 2048), (1, 0), (2, 3, 0)])
def test_empty_remap_preserves_shape_dtype_and_avoids_launch(shape):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/sparse_index_remap.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "remap_sparse_indices_triton")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    value = torch.empty(shape, dtype=torch.int32)
    assert namespace["remap_sparse_indices_triton"](value, 8, 0, 128) is value


@pytest.mark.parametrize("ranks", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("interleave", [1, 2, 64, 128, 256])
def test_bitwise_mapping_matches_python_integer_contract(ranks, interleave):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/sparse_index_remap.py"
    function = next(
        n
        for n in ast.walk(ast.parse(source.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "remap_sparse_indices_fused_kernel"
    )
    statements = [
        n
        for n in function.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("owner", "remapped", "valid")
    ]
    indices = [
        -1,
        0,
        interleave - 1,
        interleave,
        ranks * interleave - 1,
        ranks * interleave,
        2**24 - 1,
        2**24 + 127,
        2**31 - 1,
        2**40 + 255,
        2**62 - 1,
    ]
    for rank in range(ranks):
        namespace = dict(
            idx=torch.tensor(indices, dtype=torch.int64),
            dcp_rank=rank,
            DCP_BITS=ranks.bit_length() - 1,
            INTERLEAVE_BITS=interleave.bit_length() - 1,
        )
        exec(compile(ast.Module(body=statements, type_ignores=[]), str(source), "exec"), namespace)
        valid = [i >= 0 and (i // interleave) % ranks == rank for i in indices]
        mapped = [(i // (ranks * interleave)) * interleave + i % interleave for i in indices]
        assert namespace["valid"].tolist() == valid
        assert namespace["remapped"][valid].tolist() == [m for m, v in zip(mapped, valid) if v]


@pytest.mark.parametrize("ranks,interleave", [(1, 3), (3, 128), (8, 65), (2, 3), (3, 127), (5, 129)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_generic_wrapper_routes_to_exact_integer_fallback(ranks, interleave, dtype):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/sparse_index_remap.py"
    tree = ast.parse(source.read_text())
    functions: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_remap_sparse_indices_integer", "remap_sparse_indices_triton")
    ]
    # No Triton launch objects: a generic call must reach the actual fallback.
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    value = torch.tensor([-1, 0, 1, 127, 128, 2**24 - 1, 2**24, 2**24 + 1, 2**31 - 2, 2**31 - 1], dtype=dtype)
    value = value.repeat(2, 3, 1).transpose(0, 1)
    for rank in range(ranks):
        expected = torch.full_like(value, -1)
        for i in range(3):
            for j in range(2):
                owned = [int(x) for x in value[i, j] if int(x) >= 0 and (int(x) // interleave) % ranks == rank]
                mapped = [(x // (ranks * interleave)) * interleave + x % interleave for x in owned]
                expected[i, j, : len(mapped)] = torch.tensor(mapped, dtype=dtype)
        actual = namespace["remap_sparse_indices_triton"](value, ranks, rank, interleave)
        assert torch.equal(actual, expected)
