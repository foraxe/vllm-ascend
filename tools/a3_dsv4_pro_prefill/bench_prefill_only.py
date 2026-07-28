#!/usr/bin/env python3
"""Cold single-request prefill benchmark for the local vLLM OpenAI endpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import requests


def run_once(endpoint: str, words: int, timeout: float, repetition: int) -> dict[str, float | int]:
    # The changing suffix avoids accidental prefix-cache reuse if a caller enables it.
    prompt = (" hello" * words) + f" benchmark-run-{repetition}"
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 1,
        "stream": False,
    }
    started = time.perf_counter()
    response = requests.post(endpoint, json=payload, timeout=timeout)
    elapsed_s = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    usage = body.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    if prompt_tokens <= 0:
        raise RuntimeError(f"missing prompt_tokens in response: {body}")
    return {
        "repetition": repetition,
        "prompt_tokens": prompt_tokens,
        "elapsed_s": elapsed_s,
        "prefill_tok_s": prompt_tokens / elapsed_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:7100/v1/chat/completions")
    parser.add_argument("--words", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    for i in range(args.warmup):
        result = run_once(args.endpoint, args.words, args.timeout, -(i + 1))
        print("warmup", json.dumps(result, sort_keys=True), flush=True)

    results = [run_once(args.endpoint, args.words, args.timeout, i + 1) for i in range(args.runs)]
    throughputs = [float(item["prefill_tok_s"]) for item in results]
    summary = {
        "endpoint": args.endpoint,
        "requested_words": args.words,
        "warmup": args.warmup,
        "runs": args.runs,
        "samples": results,
        "median_prefill_tok_s": statistics.median(throughputs),
        "mean_prefill_tok_s": statistics.mean(throughputs),
        "min_prefill_tok_s": min(throughputs),
        "max_prefill_tok_s": max(throughputs),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

