#!/usr/bin/env python
import argparse
import json
import os
import sys
from typing import Dict, Tuple


def load_primary_scores(metrics_path: str) -> Dict[str, float]:
    scores = {}
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            task = m.get("task_name") or m.get("task_config", {}).get("task_name")
            metrics = m.get("metrics", {})
            # Only keep AGI Eval English retrieval subtasks
            if not isinstance(task, str) or not task.startswith("agi_eval_") or not task.endswith("::retrieval"):
                continue
            val = metrics.get("primary_score")
            if isinstance(val, (int, float)):
                scores[task] = float(val)
    return scores


def compute_macro(scores: Dict[str, float]) -> float:
    vals = list(scores.values())
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", required=True)
    ap.add_argument("--dir-b", required=True)
    args = ap.parse_args()

    path_a = os.path.join(args.dir_a, "metrics-all.jsonl")
    path_b = os.path.join(args.dir_b, "metrics-all.jsonl")
    if not os.path.exists(path_a):
        print(f"metrics-all.jsonl not found: {path_a}", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(path_b):
        print(f"metrics-all.jsonl not found: {path_b}", file=sys.stderr)
        sys.exit(2)

    scores_a = load_primary_scores(path_a)
    scores_b = load_primary_scores(path_b)

    keys = sorted(set(scores_a.keys()) | set(scores_b.keys()))
    print("Task, A, B, Δ(B-A)")
    for k in keys:
        a = scores_a.get(k)
        b = scores_b.get(k)
        da = f"{a:.4f}" if a is not None else "NA"
        db = f"{b:.4f}" if b is not None else "NA"
        d = "NA"
        if a is not None and b is not None:
            d = f"{(b - a):+.4f}"
        print(f"{k}, {da}, {db}, {d}")

    macro_a = compute_macro({k: v for k, v in scores_a.items() if k in keys})
    macro_b = compute_macro({k: v for k, v in scores_b.items() if k in keys})
    dmacro = macro_b - macro_a if (macro_a == macro_a and macro_b == macro_b) else float("nan")
    print("\nMacro (AGI Eval English retrieval):")
    print(f"A: {macro_a:.4f}  B: {macro_b:.4f}  Δ(B-A): {dmacro:+.4f}")


if __name__ == "__main__":
    main()


