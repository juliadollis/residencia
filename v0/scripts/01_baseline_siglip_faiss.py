# /workspace/scripts/01_baseline_siglip_faiss.py
import os, json, argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple
import yaml
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
from transformers import AutoProcessor, AutoModel

# FAISS (opcional)
try:
    import faiss  # GPU build preferred
except Exception as e:
    print("FAISS import error:", e)
    faiss = None


@dataclass
class CFG:
    dataset_id: str
    dataset_name: str               # <- "TEST" para nlphuji/flickr30k
    image_column: str               # <- "image"
    text_column: str                # <- "caption"
    caption_is_list: bool           # <- True para flickr30k (5 captions)
    split_train: str                # <- "train" (criado internamente)
    split_test: str                 # <- "test"  (criado internamente)
    model_name: str                 # <- "google/siglip-base-patch16-224"
    batch_size: int
    num_workers: int
    device: str
    out_dir: str
    faiss_index: str                # "faiss_ip" ou "faiss_l2"
    topk: int


def load_cfg(path: str) -> CFG:
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    return CFG(**y)


def seed_everything(seed=3407):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_images(batch: List[Dict], processor: AutoProcessor, image_column: str):
    images = [b[image_column] for b in batch]
    proc = processor(images=images, return_tensors="pt")
    return proc


def collate_texts(batch: List[Dict], processor: AutoProcessor, text_column: str, caption_is_list: bool):
    def pick_text(x):
        v = x[text_column]
        if caption_is_list and isinstance(v, (list, tuple)) and len(v) > 0:
            return v[0]  # usa a primeira legenda
        return v
    texts = [pick_text(b) for b in batch]
    proc = processor(text=texts, padding=True, truncation=True, return_tensors="pt")
    return proc, texts


def build_dataloaders(ds: Dict[str, Dataset], cfg: CFG, processor: AutoProcessor):
    train_loader = DataLoader(
        ds[cfg.split_train],
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        collate_fn=lambda b: (
            collate_images(b, processor, cfg.image_column),
            collate_texts(b, processor, cfg.text_column, cfg.caption_is_list),
        ),
    )
    test_loader = DataLoader(
        ds[cfg.split_test],
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        collate_fn=lambda b: (
            collate_images(b, processor, cfg.image_column),
            collate_texts(b, processor, cfg.text_column, cfg.caption_is_list),
        ),
    )
    return train_loader, test_loader


@torch.no_grad()
def embed_images(model: AutoModel, pixel_inputs: Dict[str, torch.Tensor], device: str):
    pixel_inputs = {k: v.to(device) for k, v in pixel_inputs.items()}
    out = model.get_image_features(**pixel_inputs)
    feats = torch.nn.functional.normalize(out, dim=-1)
    return feats.cpu()


@torch.no_grad()
def embed_texts(model: AutoModel, text_inputs: Dict[str, torch.Tensor], device: str):
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    out = model.get_text_features(**text_inputs)
    feats = torch.nn.functional.normalize(out, dim=-1)
    return feats.cpu()


def save_npy(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def build_faiss_index(vectors: np.ndarray, kind: str = "faiss_ip", gpu: bool = True):
    d = vectors.shape[1]
    if kind == "faiss_ip":
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)
    if gpu and torch.cuda.is_available() and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        index = faiss.index_cpu_to_all_gpus(index, co)
    index.add(vectors.astype(np.float32))
    return index


def compute_recall_at_k(sims: np.ndarray, gt: List[List[int]], ks=[1, 5, 10]):
    order = np.argsort(-sims, axis=1)  # maior similaridade primeiro
    recalls = {}
    for k in ks:
        hit = 0
        for i in range(order.shape[0]):
            topk = set(order[i, :k].tolist())
            if len(topk.intersection(set(gt[i]))) > 0:
                hit += 1
        recalls[k] = hit / order.shape[0]
    return recalls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)
    seed_everything(3407)

    os.makedirs(cfg.out_dir, exist_ok=True)

    # -------------------- Dataset --------------------
    print(">> Carregando dataset:", cfg.dataset_id, "name=", cfg.dataset_name)
    ds_raw = load_dataset(cfg.dataset_id, name=cfg.dataset_name, trust_remote_code=True)
    # o repo nlphuji/flickr30k expõe apenas o split "test" -> criamos um train/test interno
    base_split = "test" if "test" in ds_raw else list(ds_raw.keys())[0]
    raw = ds_raw[base_split]
    # 1000 amostras para avaliação, o resto para treino
    splits = raw.train_test_split(test_size=1000, seed=3407, shuffle=True)
    ds = {cfg.split_train: splits["train"], cfg.split_test: splits["test"]}
    print("Splits internos:", {k: len(v) for k, v in ds.items()})

    # -------------------- Modelo --------------------
    print(">> Carregando modelo/processor:", cfg.model_name)
    processor = AutoProcessor.from_pretrained(cfg.model_name)
    model = AutoModel.from_pretrained(cfg.model_name, torch_dtype=torch.float16).to(cfg.device)
    model.eval()

    # -------------------- Dataloaders ----------------
    print(">> Dataloaders…")
    train_loader, test_loader = build_dataloaders(ds, cfg, processor)

    # -------------------- Embeddings de Imagem -------
    def extract_image_matrix(split_name: str, loader: DataLoader):
        all_feats = []
        for (pix, _txt) in tqdm(loader, desc=f"embed_images[{split_name}]"):
            feats = embed_images(model, pix, cfg.device)
            all_feats.append(feats)
        feats = torch.cat(all_feats, dim=0).numpy()
        return feats

    print(">> Extraindo embeddings de imagens (train)…")
    img_feats_train = extract_image_matrix(cfg.split_train, train_loader)
    save_npy(os.path.join(cfg.out_dir, "img_feats_train.npy"), img_feats_train)

    print(">> Extraindo embeddings de imagens (test)…")
    img_feats_test = extract_image_matrix(cfg.split_test, test_loader)
    save_npy(os.path.join(cfg.out_dir, "img_feats_test.npy"), img_feats_test)

    # -------------------- Embeddings de Texto (test) -
    def extract_text_matrix(split_name: str, loader: DataLoader):
        all_feats = []
        all_raw = []
        for (_pix, (txt_inputs, texts)) in tqdm(loader, desc=f"embed_texts[{split_name}]"):
            feats = embed_texts(model, txt_inputs, cfg.device)
            all_feats.append(feats)
            all_raw.extend(texts)
        feats = torch.cat(all_feats, dim=0).numpy()
        return feats, all_raw

    print(">> Extraindo embeddings de textos (test)…")
    text_feats_test, raw_texts_test = extract_text_matrix(cfg.split_test, test_loader)
    save_npy(os.path.join(cfg.out_dir, "text_feats_test.npy"), text_feats_test)
    with open(os.path.join(cfg.out_dir, "texts_test.json"), "w", encoding="utf-8") as f:
        json.dump(raw_texts_test, f, ensure_ascii=False, indent=2)

    # -------------------- Baseline Retrieval ----------
    # Similaridade coseno via vetores normalizados = produto interno
    xb = img_feats_test.astype(np.float32)  # candidatos (imagens)
    xq = text_feats_test.astype(np.float32) # consultas (textos)
    sims = (xq @ xb.T)

    # ground-truth 1-a-1 (cada linha do split corresponde a 1 imagem)
    n = xq.shape[0]
    gt = [[i] for i in range(n)]
    recalls = compute_recall_at_k(sims, gt, ks=[1, 5, 10])
    print(">> Baseline (SigLIP congelado) — Recall@1/5/10:", recalls)

    # (Opcional) índice FAISS para busca rápida top-k
    if faiss is not None:
        try:
            print(">> Construindo índice FAISS…", cfg.faiss_index)
            index = build_faiss_index(xb, kind=cfg.faiss_index, gpu=True)
            D, I = index.search(xq, cfg.topk)
            # não é necessário para métricas (já temos sims), mas útil p/ depurar
            np.save(os.path.join(cfg.out_dir, "faiss_topk_idx.npy"), I)
            np.save(os.path.join(cfg.out_dir, "faiss_topk_sim.npy"), D)
        except Exception as e:
            print("FAISS indisponível durante indexação/pesquisa:", repr(e))

    # salvar métricas
    with open(os.path.join(cfg.out_dir, "baseline_metrics.json"), "w") as f:
        json.dump({"recall": recalls}, f, indent=2)

    print("✓ Concluído. Artefatos em:", cfg.out_dir)


if __name__ == "__main__":
    main()