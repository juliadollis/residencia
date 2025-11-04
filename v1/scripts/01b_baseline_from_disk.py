# /workspace/scripts/01b_baseline_from_disk.py
import os, json, argparse, io
from dataclasses import dataclass
from typing import Dict
import yaml
import numpy as np
import torch
from datasets import load_from_disk, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
from PIL import Image

@dataclass
class CFG:
    local_dataset_dir: str   # /workspace/data/flickr30k_expanded
    image_column: str        # "image"
    text_column: str         # "text"
    model_name: str          # "google/siglip-base-patch16-224"
    batch_size: int
    num_workers: int
    device: str              # "cuda"
    out_dir: str             # /workspace/experiments/...

def load_cfg(p):
    with open(p, "r") as f:
        y = yaml.safe_load(f)
    y["batch_size"]  = int(y["batch_size"])
    y["num_workers"] = int(y["num_workers"])
    return CFG(**y)

# ---- helpers ----
def _to_pil(x):
    # já é PIL
    if isinstance(x, Image.Image):
        return x
    # dataset image feature pode vir como dict {"bytes":..., "path":...}
    if isinstance(x, dict):
        b = x.get("bytes", None)
        p = x.get("path", None)
        if b is not None:
            return Image.open(io.BytesIO(b)).convert("RGB")
        if p:
            return Image.open(p).convert("RGB")
    # pode vir como caminho (string)
    if isinstance(x, str) and os.path.exists(x):
        return Image.open(x).convert("RGB")
    raise TypeError(f"Não sei converter tipo {type(x)} para PIL.Image")

def collate_images(batch, proc, col):
    images = [_to_pil(b[col]) for b in batch]
    return proc(images=images, return_tensors="pt")

def collate_texts(batch, proc, col):
    texts = [b[col] for b in batch]
    return proc(text=texts, padding=True, truncation=True, return_tensors="pt"), texts

@torch.no_grad()
def emb_img(model, pix, dev):
    x = {k: v.to(dev) for k, v in pix.items()}
    out = model.get_image_features(**x)
    return torch.nn.functional.normalize(out, dim=-1).cpu()

@torch.no_grad()
def emb_txt(model, tx, dev):
    x = {k: v.to(dev) for k, v in tx.items()}
    out = model.get_text_features(**x)
    return torch.nn.functional.normalize(out, dim=-1).cpu()

def recall_at_k(sims: np.ndarray, ks=[1, 5, 10]):
    order = np.argsort(-sims, axis=1)
    n = order.shape[0]
    out = {}
    for k in ks:
        hit = 0
        for i in range(n):
            if i in set(order[i, :k].tolist()):
                hit += 1
        out[k] = hit / n
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)

    os.makedirs(cfg.out_dir, exist_ok=True)

    print(">> load_from_disk:", cfg.local_dataset_dir)
    dsd: DatasetDict = load_from_disk(cfg.local_dataset_dir)
    train, test = dsd["train"], dsd["test"]
    print({k: len(v) for k, v in dsd.items()})

    print(">> modelo:", cfg.model_name)
    proc = AutoProcessor.from_pretrained(cfg.model_name)
    model = AutoModel.from_pretrained(cfg.model_name, dtype=torch.float16).to(cfg.device)
    model.eval()

    def mk_loader(split):
        return DataLoader(
            split,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            collate_fn=lambda b: (
                collate_images(b, proc, cfg.image_column),
                collate_texts(b, proc, cfg.text_column),
            ),
        )

    te_loader = mk_loader(test)

    print(">> embeddings test (img/text)…")
    xb_list, xq_list, raw = [], [], []
    for pix, (tx, txts) in tqdm(te_loader, desc="embed test"):
        xb_list.append(emb_img(model, pix, cfg.device))
        xq_list.append(emb_txt(model, tx, cfg.device))
        raw.extend(txts)

    xb = torch.cat(xb_list, 0).numpy().astype(np.float32)  # [Nc, d]
    xq = torch.cat(xq_list, 0).numpy().astype(np.float32)  # [Nq, d]

    np.save(os.path.join(cfg.out_dir, "img_feats_test.npy"), xb)
    np.save(os.path.join(cfg.out_dir, "text_feats_test.npy"), xq)
    with open(os.path.join(cfg.out_dir, "texts_test.json"), "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    sims = xq @ xb.T
    rec = recall_at_k(sims)
    with open(os.path.join(cfg.out_dir, "baseline_metrics.json"), "w") as f:
        json.dump({"recall": rec}, f, indent=2)

    print(">> Baseline Recall:", rec)
    print("✓ Artefatos em:", cfg.out_dir)

if __name__ == "__main__":
    main()