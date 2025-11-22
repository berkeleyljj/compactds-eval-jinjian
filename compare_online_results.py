import argparse
import json
import os
from glob import glob


def load_metrics_from_dir(out_dir: str) -> dict:
    results = {}
    # Prefer metrics-all.jsonl if present
    all_path = os.path.join(out_dir, "metrics-all.jsonl")
    if os.path.exists(all_path):
        with open(all_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                task_name = obj.get("task_config", {}).get("metadata", {}).get("alias", obj.get("task_name"))
                results[task_name] = obj.get("metrics", {})
        return results
    # Fallback: scan per-task metrics files
    for path in sorted(glob(os.path.join(out_dir, "task-*-metrics.json"))):
        with open(path, "r") as f:
            obj = json.load(f)
            task_name = obj.get("task_config", {}).get("metadata", {}).get("alias", obj.get("task_name"))
            results[task_name] = obj.get("metrics", {})
    return results


def fmt(x):
    if isinstance(x, (int, float)):
        return f"{x:.4f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diskann-dir", required=True, help="Output dir for DiskANN run")
    ap.add_argument("--ivfpq-dir", required=True, help="Output dir for IVFPQ run")
    args = ap.parse_args()

    diskann = load_metrics_from_dir(args.diskann_dir) if os.path.isdir(args.diskann_dir) else {}
    ivfpq = load_metrics_from_dir(args.ivfpq_dir) if os.path.isdir(args.ivfpq_dir) else {}

    tasks = sorted(set(diskann.keys()) | set(ivfpq.keys()))
    if not tasks:
        print("No metrics found. Check output directories.")
        print(f"DiskANN: {args.diskann_dir}")
        print(f"IVFPQ  : {args.ivfpq_dir}")
        return

    print(f"Comparing results\nDiskANN: {args.diskann_dir}\nIVFPQ  : {args.ivfpq_dir}\n")
    for t in tasks:
        dm = diskann.get(t, {})
        im = ivfpq.get(t, {})
        print(f"[{t}]")
        for key in ["f1", "exact_match", "recall", "primary_score"]:
            dval = dm.get(key, None)
            ival = im.get(key, None)
            if dval is None and ival is None:
                continue
            diff = None
            if isinstance(dval, (int, float)) and isinstance(ival, (int, float)):
                diff = ival - dval
            print(f"  {key:13s} diskann={fmt(dval)}  ivfpq={fmt(ival)}" + (f"  Δ={fmt(diff)}" if diff is not None else ""))
        print()


if __name__ == "__main__":
    main()


