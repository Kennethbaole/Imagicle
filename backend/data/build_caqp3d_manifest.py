#!/usr/bin/env python3
# Create: backend/data/build_cap3d_manifest.py
import json, os, random, pathlib
from datasets import load_dataset

OUT_DIR = "backend/data/cap3d_local"
MANIFEST = pathlib.Path(OUT_DIR) / "train_manifest.jsonl"
N_MAX = int(os.environ.get("CAP3D_MAX", "10000"))  # adjust size

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # This split name may change; check the dataset card if needed.
    ds = load_dataset("tiange/Cap3D", split="train", streaming=True)  # stream: not all to disk
    picked = 0
    with open(MANIFEST, "w") as f:
        for ex in ds:
            # Cap3D fields you’ll commonly see:
            # ex.get("objaverse_id"), ex.get("caption"), ex.get("pointcloud"), ex.get("renders") ...
            # The exact schema can evolve; print a few examples first in a REPL if unsure.
            cap = ex.get("caption")
            pcd = ex.get("pointcloud")  # usually comes as a dict with 'path' or is auto-downloaded
            if not cap or not pcd:
                continue
            # Hugging Face will cache files under ~/.cache/huggingface
            # When accessed, dataset gives you a local path in ex["pointcloud"]["path"] or similar.
            # Fall back: skip if missing
            pcd_path = None
            if isinstance(pcd, dict):
                pcd_path = pcd.get("path") or pcd.get("filename")
            elif isinstance(pcd, str):
                pcd_path = pcd
            if not pcd_path or not os.path.exists(pcd_path):
                # Trigger materialization by accessing the feature:
                # Depending on dataset config you might need to explicitly download the blob.
                continue

            rec = {
                "id": ex.get("objaverse_id"),
                "caption": cap,
                "pointcloud_path": pcd_path,
                # optional: a few render paths if available
                "render_paths": ex.get("renders", [])[:4] if ex.get("renders") else []
            }
            f.write(json.dumps(rec) + "\n")
            picked += 1
            if picked >= N_MAX:
                break
    print("Wrote", MANIFEST, "count:", picked)

if __name__ == "__main__":
    main()
