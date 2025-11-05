# workspace/v1/prepare_wikiart_desc.py
import os
import io
import argparse
from typing import List
from PIL import Image, ImageFile
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True


def save_pil(img: Image.Image, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = img.convert("RGB")
    with open(out_path, "wb") as f:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        f.write(buf.getvalue())
    return out_path


def get_texts(ex, text_field: str):
    val = ex.get(text_field, None)
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val if x is not None]
    return [str(val)]  # envelopa string


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="workspace/v1")
    ap.add_argument("--out_rel", type=str, default="data/WikiArtDesc")
    ap.add_argument("--repo", type=str, default="Artificio/WikiArt")
    ap.add_argument("--splits", type=str, default="train", help="ex.: train ou test,validation")
    ap.add_argument("--text_field", type=str, default="description")
    ap.add_argument("--limit_images", type=int, default=None)
    ap.add_argument("--max_desc_per_image", type=int, default=5)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--skip_empty", action="store_true")
    ap.add_argument("--min_desc_len", type=int, default=3)
    ap.add_argument("--images_subdir", type=str, default="images_wikiart")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = os.path.join(args.root, args.out_rel)
    os.makedirs(out_dir, exist_ok=True)
    img_root = os.path.join(args.root, args.images_subdir)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    rows_img: List[str] = []
    rows_cap: List[str] = []

    for split in splits:
        print(f">> Carregando {args.repo} split={split} …")
        ds = load_dataset(args.repo, split=split, streaming=False)
        if args.limit_images is not None:
            n = min(args.limit_images, len(ds))
            ds = ds.select(range(n))
            print(f">> limit_images={n} para split={split}")

        with_text = 0
        without_text = 0
        print(f">> Convertendo exemplos em pares (imagem, caption) – split={split}")
        for i in tqdm(range(len(ds))):
            ex = ds[i]
            texts = get_texts(ex, args.text_field)

            clean = []
            for t in texts:
                s = (t or "").strip()
                if args.skip_empty and len(s) == 0:
                    continue
                if len(s.replace(" ", "")) < args.min_desc_len:
                    continue
                clean.append(s)
            if len(clean) == 0:
                without_text += 1
                continue
            with_text += 1

            pil = ex["image"]
            if not isinstance(pil, Image.Image):
                pil = Image.fromarray(np.array(pil))

            local_name = f"wikiart_{split}_{i}.jpg"
            local_path = os.path.join(img_root, local_name)
            if not os.path.exists(local_path):
                try:
                    save_pil(pil, local_path)
                except Exception:
                    continue

            if args.max_desc_per_image is not None:
                clean = clean[: args.max_desc_per_image]
            for cap in clean:
                rows_img.append(local_path)
                rows_cap.append(cap)

        print(f">> Split {split}: com_texto={with_text}, sem_texto={without_text}")

    print(f">> Total de pares após expansão: {len(rows_img)}")
    if len(rows_img) == 0:
        raise RuntimeError("Nenhum par gerado. Verifique --text_field e splits.")

    idx_all = np.arange(len(rows_img))
    if args.val_ratio > 0:
        tr_idx, va_idx = train_test_split(
            idx_all, test_size=args.val_ratio, random_state=args.seed, shuffle=True
        )
    else:
        tr_idx, va_idx = idx_all, np.array([], dtype=int)

    def make_ds(idxs: np.ndarray) -> Dataset:
        return Dataset.from_dict({
            "image_path": [rows_img[j] for j in idxs],
            "caption":    [rows_cap[j] for j in idxs],
        })

    ds_train = make_ds(tr_idx)
    ds_val   = make_ds(va_idx)
    dsd = DatasetDict({"train": ds_train, "validation": ds_val})
    dsd.save_to_disk(out_dir)
    print(f">> Dataset salvo em: {out_dir}")
    print({k: len(v) for k, v in dsd.items()})
    if len(ds_train) > 0:
        print(">> Exemplo:", {k: ds_train[0][k] for k in ds_train.features})


if __name__ == "__main__":
    main()
