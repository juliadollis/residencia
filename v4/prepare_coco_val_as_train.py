import os
import argparse
from datasets import load_dataset, DatasetDict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="workspace/v1")
    ap.add_argument("--repo", type=str, default="lmms-lab/COCO-Caption")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--max_samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    out_dir = os.path.join(args.root, "data", "COCOValMini")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset(args.repo, split=args.split, streaming=False)
    n = len(ds)
    m = min(args.max_samples, n)
    ds = ds.select(range(m))

    def _pick_caption(ex):
        caps = ex.get("answer", [])
        cap = caps[0] if isinstance(caps, list) and len(caps) > 0 else ""
        url = ex.get("coco_url", None)
        if not url:
            url = ex.get("image", None)
        return {"image_path": url, "caption": cap}

    cols_to_remove = [c for c in ds.column_names if c not in ["image", "coco_url", "answer"]]
    ds = ds.map(_pick_caption, remove_columns=cols_to_remove, desc="normalize_coco")

    idx = list(range(len(ds)))
    import random
    rng = random.Random(args.seed)
    rng.shuffle(idx)
    v = int(len(idx) * args.val_ratio)
    val_idx = sorted(idx[:v])
    train_idx = sorted(idx[v:])
    train_ds = ds.select(train_idx)
    val_ds = ds.select(val_idx)

    DatasetDict({"train": train_ds, "validation": val_ds}).save_to_disk(out_dir)

if __name__ == "__main__":
    main()
