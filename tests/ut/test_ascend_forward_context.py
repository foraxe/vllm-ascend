# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend import ascend_forward_context as forward_context
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.utils import AscendDeviceType


def _moe_ep_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(quantize="w4a8")),
        parallel_config=SimpleNamespace(enable_expert_parallel=True),
    )


@patch("vllm_ascend.ascend_forward_context.get_ep_group")
@patch("vllm_ascend.ascend_forward_context.get_ascend_config")
@patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A3)
@patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=512)
@patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True)
def test_a3_prefill_selects_fused_mc2_above_mc2_capacity(
    _is_moe, _capacity, _device_type, get_ascend_config, get_ep_group
):
    """The 8K prefill A/B cannot silently fall back to AllToAllV."""
    get_ep_group.return_value = MagicMock(world_size=16)
    get_ascend_config.return_value = SimpleNamespace(enable_fused_mc2=1)

    comm = forward_context.select_moe_comm_method(8192, _moe_ep_config())

    assert comm is MoECommType.FUSED_MC2


@patch("vllm_ascend.ascend_forward_context.get_ep_group")
@patch("vllm_ascend.ascend_forward_context.get_ascend_config")
@patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A3)
@patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=512)
@patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True)
def test_a3_prefill_uses_alltoall_when_fused_mc2_is_disabled(
    _is_moe, _capacity, _device_type, get_ascend_config, get_ep_group
):
    get_ep_group.return_value = MagicMock(world_size=16)
    get_ascend_config.return_value = SimpleNamespace(enable_fused_mc2=0)

    comm = forward_context.select_moe_comm_method(8192, _moe_ep_config())

    assert comm is MoECommType.ALLTOALL
