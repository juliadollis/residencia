import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import argparse
from datasets import load_from_disk
from transformers import AutoProcessor, AutoModel
from workspace.v1.utils_image_cache import ImageFetcher

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="workspace/v1")
    ap.add_argument("--data_rel", type=str, default="data/COCOValMini")
    ap.add_argument("--model", type=str, default="google/siglip-base-patch16-224")
    args = ap.parse_args()

    ds = load_from_disk(os.path.join(args.root, args.data_rel))["validation"]
    n = min(64, len(ds))
    caps = [ds[i]["caption"] for i in range(n)]
    imgs = [ds[i]["image_path"] for i in range(n)]
    fetcher = ImageFetcher(cache_dir=os.path.join(args.root, "cache", "images"))

    ok = 0
    for p in imgs:
        try:
            _ = fetcher.fetch_pil(p)
            ok += 1
        except:
            pass

    _ = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    _ = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    print(json.dumps({"ok_images": ok, "sampled": n}, ensure_ascii=False))

if __name__ == "__main__":
    main()
