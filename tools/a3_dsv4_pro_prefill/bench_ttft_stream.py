#!/usr/bin/env python3
"""Fixed-shape, cold streaming TTFT benchmark for a local OpenAI endpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests


def _has_text_delta(event: dict[str, Any]) -> bool:
    for choice in event.get("choices", []):
        delta = choice.get("delta", {})
        if any(delta.get(field) for field in ("content", "reasoning_content")):
            return True
    return False


def _percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


def run_once(endpoint: str, words: int, timeout: float, repetition: int) -> dict[str, float | int]:
    # The changing suffix prevents accidental prefix-cache reuse if configuration drifts.
    prompt = (" hello" * words) + f" benchmark-run-{repetition}"
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_sse_s: float | None = None
    first_token_s: float | None = None
    completion = ""
    response = requests.post(endpoint, json=payload, timeout=timeout, stream=True)
    response.raise_for_status()
    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        now = time.perf_counter()
        data = line[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        if first_sse_s is None:
            first_sse_s = now - started
        if _has_text_delta(event):
            if first_token_s is None:
                first_token_s = now - started
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                completion += delta.get("content") or delta.get("reasoning_content") or ""
    elapsed_s = time.perf_counter() - started
    if first_sse_s is None or first_token_s is None:
        raise RuntimeError("stream ended without an SSE event containing a text token")
    return {
        "repetition": repetition,
        "first_sse_s": first_sse_s,
        "first_token_s": first_token_s,
        "elapsed_s": elapsed_s,
        "completion_chars": len(completion),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:7100/v1/chat/completions")
    parser.add_argument("--words", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    for i in range(args.warmup):
        print("warmup", run_once(args.endpoint, args.words, args.timeout, -(i + 1)), flush=True)
    samples = [run_once(args.endpoint, args.words, args.timeout, i + 1) for i in range(args.runs)]
    ttft = [float(sample["first_token_s"]) for sample in samples]
    first_sse = [float(sample["first_sse_s"]) for sample in samples]
    summary = {
        "metric": "client first nonempty text-delta TTFT",
        "endpoint": args.endpoint,
        "requested_words": args.words,
        "warmup": args.warmup,
        "runs": args.runs,
        "samples": samples,
        "median_ttft_s": statistics.median(ttft),
        "p90_ttft_s": _percentile(ttft, 90),
        "mean_ttft_s": statistics.mean(ttft),
        "median_first_sse_s": statistics.median(first_sse),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

