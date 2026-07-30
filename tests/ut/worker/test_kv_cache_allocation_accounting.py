#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import torch

from vllm_ascend.worker.kv_cache_allocation_accounting import (
    summarize_kv_cache_allocations,
)


def test_counts_backing_storage_instead_of_aliased_tensor_views() -> None:
    baseline_storage = torch.empty(128, dtype=torch.uint8)
    baseline_view = baseline_storage[16:48]
    compact_owner = torch.empty(24, dtype=torch.float32)
    shared_stage = torch.empty((8, 16), dtype=torch.float16)

    summary = summarize_kv_cache_allocations(
        baseline_raw=[
            baseline_view,
            baseline_storage[48:64],
            {"shared_layer": baseline_view},
        ],
        compact_owner=[
            (compact_owner.view(6, 4),),
            compact_owner[8:16],
        ],
        c128_stages=[
            shared_stage.view(-1),
            shared_stage[:, :8],
            [shared_stage[2:6]],
        ],
    )

    assert summary.baseline_raw_bytes == 128
    assert summary.compact_owner_bytes == 96
    assert summary.c128_stage_bytes == 256
    assert summary.total_unique_bytes == 480
    assert summary.unique_storages == 3
    assert summary.duplicate_references == 5
    assert summary.cross_category_aliases == 0


def test_cross_category_alias_is_counted_once_in_specific_category() -> None:
    reused_storage = torch.empty(64, dtype=torch.uint8)

    summary = summarize_kv_cache_allocations(
        baseline_raw=[reused_storage],
        compact_owner=[reused_storage.view(8, 8)],
        c128_stages=[reused_storage[:16]],
    )

    assert summary.baseline_raw_bytes == 0
    assert summary.compact_owner_bytes == 64
    assert summary.c128_stage_bytes == 0
    assert summary.total_unique_bytes == 64
    assert summary.unique_storages == 1
    assert summary.duplicate_references == 2
    assert summary.cross_category_aliases == 1


def test_empty_categories_and_log_line_are_stable() -> None:
    summary = summarize_kv_cache_allocations(
        baseline_raw=[],
        compact_owner=[],
        c128_stages=[],
    )

    assert summary.format_log_line(rank=7, num_blocks=4190) == (
        "KV_CACHE_ALLOCATION rank=7 num_blocks=4190"
        " baseline_raw_bytes=0 compact_owner_bytes=0"
        " c128_stage_bytes=0 total_unique_bytes=0"
        " unique_storages=0 duplicate_references=0"
        " cross_category_aliases=0"
    )
