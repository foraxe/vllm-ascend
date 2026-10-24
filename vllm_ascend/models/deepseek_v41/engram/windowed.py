# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Submit late-layer BF16 WKV pieces at attention/FFN boundaries."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from .embedding import AscendParallelEngramEmbedding
from .layer import AscendEngram


class WindowedEngramProjection:
    """One forward's scheduling state; buffers/streams/events belong to the model.

    Python counters are used while building eager work or capturing a graph.
    Replay executes the captured pieces and dependencies at fixed addresses.
    """

    def __init__(
        self,
        layer: AscendEngram,
        rows: torch.Tensor,
        output: torch.Tensor,
        stream,
        events: Sequence,
        *,
        first_layer: int,
        consumer_layer: int,
        tokens: int,
        shard: Callable[[torch.Tensor], torch.Tensor],
        lookup_ready=None,
        external: bool = False,
    ):
        self.layer, self.rows, self.output = layer, rows, output
        self.stream, self.events = stream, events
        self.first_layer, self.consumer_layer = first_layer, consumer_layer
        self.tokens, self.shard = tokens, shard
        self.lookup_ready, self.external = lookup_ready, external
        self.chunks = tuple(range(0, output.shape[1], layer.dim))
        windows = consumer_layer - first_layer
        if len(self.chunks) > windows or len(events) < len(self.chunks):
            raise ValueError("Engram projection needs at least one window per key/value piece")
        self.schedule = {i * windows // len(self.chunks): i for i in range(len(self.chunks))}
        self.next_chunk = 0
        self.joined = False
        self.has_work = False

    @property
    def started(self) -> bool:
        return self.next_chunk > 0

    def submit_window(self, index: int) -> None:
        """Offer one attention-output boundary; skipped windows are harmless."""
        if self.schedule.get(index) == self.next_chunk:
            self._submit(self.events[self.next_chunk])

    def _submit(self, event) -> None:
        self.has_work = True
        main = torch.npu.current_stream()
        event.record(main)
        self.stream.wait_event(event)
        with torch.npu.stream(self.stream):
            if not self.started:
                if self.lookup_ready is not None:
                    AscendParallelEngramEmbedding.wait_lookup(self.lookup_ready, external=self.external)
                self.rows = self.shard(self.rows)[: self.tokens]
                self.rows.record_stream(self.stream)
                self.output.record_stream(self.stream)
            start = self.chunks[self.next_chunk]
            stop = min(start + self.layer.dim, self.output.shape[1])
            self.layer.project_chunk_into(self.rows, self.output[: self.tokens, start:stop], start, stop)
        self.next_chunk += 1

    def finish(self) -> torch.Tensor:
        """Complete any pieces whose windows were skipped before consuming KV."""
        while self.next_chunk < len(self.chunks):
            self._submit(self.events[self.next_chunk])
        self.close()
        return self.output[: self.tokens]

    def close(self) -> None:
        if self.has_work and not self.joined:
            torch.npu.current_stream().wait_stream(self.stream)
            self.joined = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
