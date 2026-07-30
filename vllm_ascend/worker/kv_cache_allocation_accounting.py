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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

_CATEGORY_PRIORITY = (
    "compact_owner",
    "c128_stage",
    "baseline_raw",
)


@dataclass(frozen=True)
class KVCacheAllocationSummary:
    """Physical backing-storage accounting for one model-runner rank."""

    baseline_raw_bytes: int
    compact_owner_bytes: int
    c128_stage_bytes: int
    total_unique_bytes: int
    unique_storages: int
    duplicate_references: int
    cross_category_aliases: int

    def format_log_line(self, *, rank: int, num_blocks: int) -> str:
        """Return one parseable, bounded log record."""
        return (
            "KV_CACHE_ALLOCATION"
            f" rank={rank}"
            f" num_blocks={num_blocks}"
            f" baseline_raw_bytes={self.baseline_raw_bytes}"
            f" compact_owner_bytes={self.compact_owner_bytes}"
            f" c128_stage_bytes={self.c128_stage_bytes}"
            f" total_unique_bytes={self.total_unique_bytes}"
            f" unique_storages={self.unique_storages}"
            f" duplicate_references={self.duplicate_references}"
            f" cross_category_aliases={self.cross_category_aliases}"
        )


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_tensors(child)
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _iter_tensors(child)


def _storage_key(tensor: torch.Tensor) -> tuple[str, int | None, int, int]:
    storage = tensor.untyped_storage()
    return (
        tensor.device.type,
        tensor.device.index,
        storage.data_ptr(),
        storage.nbytes(),
    )


def summarize_kv_cache_allocations(
    *,
    baseline_raw: Iterable[Any],
    compact_owner: Iterable[Any],
    c128_stages: Iterable[Any],
) -> KVCacheAllocationSummary:
    """Count unique physical storages across KV allocation categories.

    A storage seen through several tensor views or layer aliases is counted
    exactly once by its device, backing pointer, and backing byte size. If a
    storage crosses categories, the most specific category wins according to
    ``compact_owner``, ``c128_stage``, then ``baseline_raw``. Such overlaps are
    also reported because they normally indicate a categorization bug.

    This function only inspects tensor metadata. It performs no tensor
    operations, device reads, or synchronization. The result measures
    PyTorch-visible backing storage, not allocator reservation granularity or
    the committed physical pages behind a sparse VMM mapping.
    """
    category_values = {
        "baseline_raw": baseline_raw,
        "compact_owner": compact_owner,
        "c128_stage": c128_stages,
    }
    storage_categories: dict[tuple[str, int | None, int, int], set[str]] = {}
    tensor_references = 0

    for category, values in category_values.items():
        for tensor in _iter_tensors(values):
            tensor_references += 1
            storage_categories.setdefault(_storage_key(tensor), set()).add(category)

    category_bytes = dict.fromkeys(_CATEGORY_PRIORITY, 0)
    cross_category_aliases = 0
    for storage_key, categories in storage_categories.items():
        if len(categories) > 1:
            cross_category_aliases += 1
        assigned_category = next(category for category in _CATEGORY_PRIORITY if category in categories)
        category_bytes[assigned_category] += storage_key[-1]

    total_unique_bytes = sum(category_bytes.values())
    unique_storages = len(storage_categories)
    return KVCacheAllocationSummary(
        baseline_raw_bytes=category_bytes["baseline_raw"],
        compact_owner_bytes=category_bytes["compact_owner"],
        c128_stage_bytes=category_bytes["c128_stage"],
        total_unique_bytes=total_unique_bytes,
        unique_storages=unique_storages,
        duplicate_references=tensor_references - unique_storages,
        cross_category_aliases=cross_category_aliases,
    )
