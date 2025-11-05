# workspace/v1/check_alignment.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse, numpy as np, torch
from datasets import load_from_disk
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
from workspace.v1.utils_image_cache import ImageFetcher

@torch.no_grad()
def encode(proc, model, device, ds, max_len=64, bs_img=64, bs_txt=128, fetcher=None):
    pils, caps = [], []
    for i in range(len(ds)):
        pils.append(fetcher.fetch_pil(ds[i]["image_path"]))
        caps.append(ds[i]["caption"])
    imgs_vec = []; 
    for i in tqdm(range(0, len(pils), bs_img), desc="img"):
        inputs = proc(images=pils[i:i+bs_img], return_tensors="pt").to(device)
        v = model.get_image_features(**inputs); v = torch.nn.functional.normalize(v, dim=-1)
        imgs_vec.append(v.cpu())
    imgs_vec = torch.cat(imgs_vec, dim=0)
    txt_vec = []
    for i in tqdm(range(0, len(caps), bs_txt), desc="txt"):
        inputs = proc(text=caps[i:i+bs_txt], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        t = model.get_text_features(**inputs); t = torch.nn.functional.normalize(t, dim=-1)
        txt_vec.append(t.cpu())
    txt_vec = torch.cat(txt_vec, dim=0)
    return imgs_vec.numpy(), txt_vec.numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="workspace/v1")
    ap.add_argument("--data_rel", type=str, default="data/WikiArtDesc")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--model", type=str, default="google/siglip-base-patch16-224")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--bs_img", type=int, default=64)
    ap.add_argument("--bs_txt", type=int, default=128)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = load_from_disk(os.path.join(args.root, args.data_rel))[args.split]
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    fetcher = ImageFetcher(cache_dir=os.path.join(args.root, "cache", "images"))

    V, T = encode(proc, model, device, ds, max_len=args.max_len, bs_img=args.bs_img, bs_txt=args.bs_txt, fetcher=fetcher)
    S = T @ V.T  # [N,N]
    n = S.shape[0]
    ranks = (-S).argsort(axis=1)
    diag_hits_at1 = float(np.mean(ranks[:,0] == np.arange(n)))
    diag_mean_pos = float(np.mean([np.where(ranks[i]==i)[0][0] for i in range(n)]))
    diag_vals = S[np.arange(n), np.arange(n)]
    off_vals = S.copy(); off_vals[np.arange(n), np.arange(n)] = np.nan
    off_mean = float(np.nanmean(off_vals))
    print({"diag_hits@1": diag_hits_at1, "diag_mean_pos": diag_mean_pos,
           "diag_mean_sim": float(np.mean(diag_vals)), "offdiag_mean_sim": off_mean})

if __name__ == "__main__":
    main()
