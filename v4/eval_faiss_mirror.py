# workspace/v1/eval_faiss_mirror.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse, numpy as np, torch
from datasets import load_from_disk, load_dataset
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
from PIL import Image
from workspace.v1.utils_image_cache import ImageFetcher

try:
    import faiss
except Exception:
    raise RuntimeError("Instale o FAISS: pip install faiss-cpu  (ou faiss-gpu)")

def compute_metrics(I, gt, k_list=(1,5,10)):
    recalls = {k: [] for k in k_list}; mrrs, ndcgs = [], []
    for qi, g in enumerate(gt):
        ranked = I[qi]; hits = (ranked == g).astype(int)
        for k in k_list: recalls[k].append(int(g in ranked[:k]))
        where = np.where(hits==1)[0]; mrrs.append(1.0/(where[0]+1) if len(where) else 0.0)
        gains = hits[:10]; dcg = np.sum(gains/np.log2(np.arange(1,len(gains)+1)+1)); ndcgs.append(dcg/1.0)
    out = {f"recall@{k}": float(np.mean(recalls[k])) for k in k_list}
    out["mrr"] = float(np.mean(mrrs)); out["ndcg@10"] = float(np.mean(ndcgs))
    return out

@torch.no_grad()
def enc_images(proc, model, device, pils, bs=64):
    vecs = []
    for i in tqdm(range(0, len(pils), bs), desc="encode_images"):
        inputs = proc(images=pils[i:i+bs], return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        vecs.append(feats.cpu())
    return torch.cat(vecs, dim=0).numpy()

@torch.no_grad()
def enc_texts(proc, model, device, texts, bs=128, max_len=64):
    vecs = []
    for i in tqdm(range(0, len(texts), bs), desc="encode_texts"):
        inputs = proc(text=texts[i:i+bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        feats = model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        vecs.append(feats.cpu())
    return torch.cat(vecs, dim=0).numpy()

def eval_disk(root, data_rel, split, model_id, device, limit=None, max_len=64, bs_img=64, bs_txt=128):
    dsd = load_from_disk(os.path.join(root, data_rel)); ds = dsd[split]
    if limit is not None: ds = ds.select(range(min(limit, len(ds))))
    processor = AutoProcessor.from_pretrained(model_id); model = AutoModel.from_pretrained(model_id).to(device).eval()
    caps = [ds[i]["caption"] for i in range(len(ds))]; paths = [ds[i]["image_path"] for i in range(len(ds))]
    fetcher = ImageFetcher(cache_dir=os.path.join(root, "cache", "images"))
    pils = [fetcher.fetch_pil(p) for p in paths]
    img_vecs = enc_images(processor, model, device, pils, bs=bs_img)
    txt_vecs = enc_texts(processor, model, device, caps, bs=bs_txt, max_len=max_len)
    d = img_vecs.shape[1]; index = faiss.IndexFlatIP(d); index.add(img_vecs); D,I = index.search(txt_vecs, 10)
    gt = list(range(len(ds))); return compute_metrics(I, gt, k_list=[1,5,10])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["disk"], default="disk")
    ap.add_argument("--root", type=str, default="workspace/v1")
    ap.add_argument("--data_rel", type=str, default="data/WikiArtDesc")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--model", type=str, default="google/siglip-base-patch16-224")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--bs_img", type=int, default=64)
    ap.add_argument("--bs_txt", type=int, default=128)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = eval_disk(args.root, args.data_rel, args.split, args.model, device,
                    limit=args.limit, max_len=args.max_len, bs_img=args.bs_img, bs_txt=args.bs_txt)
    print(out)

if __name__ == "__main__":
    main()
