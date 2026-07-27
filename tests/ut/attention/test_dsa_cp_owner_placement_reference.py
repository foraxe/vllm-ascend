# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""CPU reference gates for the DSA-CP owner-compute design.

These tests intentionally model data ownership and cache placement only. They
do not claim that a Python loop is a distributed/NPU implementation. A later
HCCL or VMM transport must match this oracle before it can replace the current
hidden-state AllGather path.
"""

import pytest
import torch

pytestmark = pytest.mark.cpu_test


def _place_kv(cache: torch.Tensor, slots: torch.Tensor, kv: torch.Tensor) -> None:
    if slots.ndim != 1 or kv.ndim != 2 or slots.numel() != kv.shape[0]:
        raise ValueError("slots must be 1-D and match the KV token dimension")
    if slots.unique().numel() != slots.numel():
        raise ValueError("one direct-placement operation cannot write a slot twice")
    cache[:, slots] = kv.unsqueeze(0)


@pytest.mark.parametrize("world_size", [2, 4])
def test_owner_wkv_direct_placement_matches_allgather_baseline(world_size: int) -> None:
    """One owner WKV per sequence shard gives the same replicated KV cache.

    This is the first DSA-CP semantic gate: compute WKV only at the source
    token owner, then place that produced KV at every required local cache.
    It replaces an AllGather of hidden states followed by repeated WKV.
    """
    torch.manual_seed(7)
    tokens_per_owner, hidden_dim, kv_dim = 3, 5, 4
    total_tokens = world_size * tokens_per_owner
    hidden = torch.randn(total_tokens, hidden_dim)
    wkv = torch.randn(hidden_dim, kv_dim)
    slots = torch.randperm(total_tokens, generator=torch.Generator().manual_seed(11))

    baseline = torch.zeros(world_size, total_tokens, kv_dim)
    _place_kv(baseline, slots, hidden @ wkv)

    owner_direct = torch.zeros_like(baseline)
    for owner in range(world_size):
        start = owner * tokens_per_owner
        end = start + tokens_per_owner
        owner_kv = hidden[start:end] @ wkv
        _place_kv(owner_direct, slots[start:end], owner_kv)

    torch.testing.assert_close(owner_direct, baseline)


def test_direct_placement_rejects_colliding_slots() -> None:
    cache = torch.zeros(2, 8, 3)
    with pytest.raises(ValueError, match="write a slot twice"):
        _place_kv(cache, torch.tensor([2, 2]), torch.ones(2, 3))


def _stateful_compressor(tokens: torch.Tensor, initial_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal ordered recurrence standing in for compressor state dependence."""
    state = initial_state.clone()
    outputs = []
    for token in tokens:
        state = state * 2 + token
        outputs.append(state)
    return torch.stack(outputs), state


def test_stateful_compressor_needs_prefix_state_handoff() -> None:
    """C4/C128 state cannot use the stateless owner-placement shortcut."""
    tokens = torch.tensor([1.0, 2.0, 3.0, 4.0])
    global_output, _ = _stateful_compressor(tokens, torch.tensor(0.0))

    first_output, first_final_state = _stateful_compressor(tokens[:2], torch.tensor(0.0))
    naive_second_output, _ = _stateful_compressor(tokens[2:], torch.tensor(0.0))
    handed_off_second_output, _ = _stateful_compressor(tokens[2:], first_final_state)

    assert not torch.equal(torch.cat([first_output, naive_second_output]), global_output)
    torch.testing.assert_close(torch.cat([first_output, handed_off_second_output]), global_output)
