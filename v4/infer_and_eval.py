# workspace/v1/infer_and_eval.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
from workspace.v1.utils_image_cache import ImageFetcher
import csv

def parse_k_list(s):
    if not s: return [1,5,10,50]
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def build_lists(ds):
    caps = [ds[i]["caption"] for i in range(len(ds))]
    imgs = [ds[i]["image_path"] for i in range(len(ds))]
    return caps, imgs

def batchify(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs], i

def get_text_cls(text_model, input_ids, attention_mask):
    out = text_model(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    return out.last_hidden_state[:, 0, :]

def inject_after_cls(vision_model, pixel_values, prompt_tokens):
    target_dtype = next(vision_model.parameters()).dtype
    x = vision_model.embeddings(pixel_values=pixel_values.to(dtype=target_dtype))
    x = x[0] if isinstance(x, tuple) else x
    prompt_tokens = prompt_tokens.to(x.dtype)
    cls, rest = x[:, :1, :], x[:, 1:, :]
    x = torch.cat([cls, prompt_tokens, rest], dim=1)
    h = vision_model.encoder(inputs_embeds=x)[0]
    v = vision_model.post_layernorm(h[:, 0, :])
    return v

class TextGuidedPromptMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_tokens, token_dim, init_std=0.01):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, out_tokens * token_dim)
        self.out_tokens = out_tokens; self.token_dim = token_dim
        with torch.no_grad():
            for m in [self.fc1, self.fc2, self.fc3]:
                torch.nn.init.trunc_normal_(m.weight, std=init_std)
                torch.nn.init.zeros_(m.bias)
    def forward(self, txt_cls):
        x = self.act(self.fc1(txt_cls))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x.view(x.size(0), self.out_tokens, self.token_dim)

def compute_baseline_features(base, proc, device, fetcher, images, captions,
                              batch_img, batch_txt, max_len, amp_dtype):
    all_img = []
    for chunk, _ in tqdm(batchify(images, batch_img),
                         total=(len(images)+batch_img-1)//batch_img,
                         desc="Baseline imagens"):
        ims = [fetcher.fetch_pil(p) for p in chunk]
        enc = proc(images=ims, return_tensors="pt")
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        with torch.amp.autocast("cuda", enabled=(device.type=="cuda"), dtype=amp_dtype):
            v = base.get_image_features(pixel_values=enc["pixel_values"])
            v = torch.nn.functional.normalize(v, dim=-1)
        all_img.append(v.detach().cpu())
    all_img = torch.cat(all_img, dim=0)

    all_txt = []
    for chunk, _ in tqdm(batchify(captions, batch_txt),
                         total=(len(captions)+batch_txt-1)//batch_txt,
                         desc="Baseline textos"):
        enc = proc(text=chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=device)
        with torch.amp.autocast("cuda", enabled=(device.type=="cuda"), dtype=amp_dtype):
            t = base.get_text_features(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            t = torch.nn.functional.normalize(t, dim=-1)
        all_txt.append(t.detach().cpu())
    all_txt = torch.cat(all_txt, dim=0)
    return all_img, all_txt

def recall_at_k(scores, k_list):
    n = scores.shape[0]
    ranks = (-scores).argsort(axis=1)
    target = np.arange(n)
    out = {}
    for k in k_list:
        hits = 0
        for i in range(n):
            pos = int(np.where(ranks[i] == target[i])[0][0])
            if pos < k: hits += 1
        out[k] = hits / n
    return out

def mean_ap_single_positive(scores):
    n = scores.shape[0]
    ranks = (-scores).argsort(axis=1)
    target = np.arange(n)
    ap = []
    for i in range(n):
        pos = int(np.where(ranks[i] == target[i])[0][0])
        ap.append(1.0 / (pos + 1))
    return float(np.mean(ap))

def maybe_save_csv(path, captions, images, scores, topk):
    n = scores.shape[0]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_index","caption","rank_of_positive","image_index","image_path","score","is_positive"])
        ranks = (-scores).argsort(axis=1); target = np.arange(n)
        for i in range(n):
            pos = int(np.where(ranks[i] == target[i])[0][0])
            top_idx = ranks[i][:topk]
            for j in top_idx:
                w.writerow([i, captions[i], pos, int(j), images[int(j)], float(scores[i,j]), int(j==i)])

def eval_baseline(base, proc, device, fetcher, ds, batch_img, batch_txt, max_len, amp_dtype, k_list, save_csv):
    captions, images = build_lists(ds)
    img_feat, txt_feat = compute_baseline_features(base, proc, device, fetcher, images, captions,
                                                   batch_img, batch_txt, max_len, amp_dtype)
    sims = (txt_feat @ img_feat.T).cpu().numpy()
    rec = recall_at_k(sims, k_list)
    map1 = mean_ap_single_positive(sims)
    if save_csv: maybe_save_csv(save_csv, captions, images, sims, topk=max(k_list))
    return rec, map1

def load_mlp(ckpt_path, txt_dim, img_dim, device, dtype):
    sd = torch.load(ckpt_path, map_location="cpu")
    cfg = sd["cfg"]
    mlp = TextGuidedPromptMLP(txt_dim, cfg["hidden_dim"], cfg["prompt_tokens"], img_dim, init_std=0.01).to(device=device, dtype=dtype)
    mlp.load_state_dict(sd["mlp"], strict=True); mlp.eval()
    return mlp, cfg

def eval_elip(base, proc, device, fetcher, ds, ckpt_path, batch_img, batch_txt, max_len, amp_dtype, k_list, save_csv):
    n = len(ds); captions, images = build_lists(ds)
    img_dim = base.config.vision_config.hidden_size
    txt_dim = base.config.text_config.hidden_size
    dtype = next(base.parameters()).dtype
    mlp, _ = load_mlp(ckpt_path, txt_dim, img_dim, device, dtype)

    # pré-encode textos
    txt_feat_all = []
    for chunk, _ in tqdm(batchify(captions, batch_txt),
                         total=(len(captions)+batch_txt-1)//batch_txt,
                         desc="ELIP textos"):
        enc = proc(text=chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=device)
        with torch.amp.autocast("cuda", enabled=(device.type=="cuda"), dtype=amp_dtype):
            t = base.get_text_features(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            t = torch.nn.functional.normalize(t, dim=-1)
        txt_feat_all.append(t.detach().cpu())
    txt_feat_all = torch.cat(txt_feat_all, dim=0)

    sims_full = np.zeros((n, n), dtype=np.float32)
    for qi in tqdm(range(n), desc="ELIP t->i"):
        q = captions[qi]
        enc_txt = proc(text=[q], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc_txt = {k: v.to(device, non_blocking=True) for k, v in enc_txt.items()}
        if "attention_mask" not in enc_txt:
            enc_txt["attention_mask"] = torch.ones_like(enc_txt["input_ids"], device=device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type=="cuda"), dtype=amp_dtype):
                tcls = get_text_cls(base.text_model, enc_txt["input_ids"], enc_txt["attention_mask"])
                prompts = mlp(tcls.to(next(mlp.parameters()).dtype))

        sims_row = []
        for chunk, _ in batchify(images, batch_img):
            ims = [fetcher.fetch_pil(p) for p in chunk]
            enc_im = proc(images=ims, return_tensors="pt")
            enc_im = {k: v.to(device, non_blocking=True) for k, v in enc_im.items()}
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device.type=="cuda"), dtype=amp_dtype):
                    vfeat = inject_after_cls(base.vision_model, enc_im["pixel_values"],
                                             prompts.repeat(enc_im["pixel_values"].size(0), 1, 1))
                    vfeat = torch.nn.functional.normalize(vfeat, dim=-1)
            tfeat_q = txt_feat_all[qi:qi+1].to(vfeat.device)
            s = torch.sum(vfeat * tfeat_q, dim=-1)
            sims_row.append(s.detach().cpu())
        sims_row = torch.cat(sims_row, dim=0).numpy()
        sims_full[qi] = sims_row

    rec = recall_at_k(sims_full, k_list)
    map1 = mean_ap_single_positive(sims_full)
    if save_csv: maybe_save_csv(save_csv, captions, images, sims_full, topk=max(k_list))
    return rec, map1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="workspace/v1")
    ap.add_argument("--data_rel", type=str, default="data/WikiArtDesc")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--model", type=str, default="google/siglip-base-patch16-224")
    ap.add_argument("--ckpt_rel", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_img", type=int, default=64)
    ap.add_argument("--batch_txt", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--k_list", type=str, default="1,5,10,50")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--save_csv_baseline", type=str, default="")
    ap.add_argument("--save_csv_elip", type=str, default="")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    data_dir = os.path.join(args.root, args.data_rel)
    dsd = load_from_disk(data_dir)
    ds = dsd[args.split]
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f">> Limitando avaliação a {len(ds)} amostras")

    fetcher = ImageFetcher(cache_dir=os.path.join(args.root, "cache", "images"), timeout=15, max_retries=3)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if device.type == "cuda" else torch.float32)
    base = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype).to(device)
    base.eval()
    for p in base.parameters(): p.requires_grad = False

    k_list = parse_k_list(args.k_list)

    print(">> Avaliando baseline")
    rec_b, map_b = eval_baseline(base, proc, device, fetcher, ds,
                                 args.batch_img, args.batch_txt, args.max_len, amp_dtype, k_list,
                                 args.save_csv_baseline if args.save_csv_baseline else None)
    print({f"Baseline R@{k}": round(rec_b[k], 4) for k in k_list})
    print({"Baseline mAP": round(map_b, 4)})

    if args.ckpt_rel:
        ckpt_path = os.path.join(args.root, args.ckpt_rel)
        if os.path.isdir(ckpt_path):
            cand_best = os.path.join(ckpt_path, "best.pt")
            cand_last = os.path.join(ckpt_path, "last.pt")
            if os.path.isfile(cand_best): ckpt_path = cand_best
            elif os.path.isfile(cand_last): ckpt_path = cand_last
            else: raise FileNotFoundError(f"Nenhum best.pt/last.pt em {ckpt_path}")
        elif not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")

        print(">> Avaliando ELIP-S")
        rec_e, map_e = eval_elip(base, proc, device, fetcher, ds, ckpt_path,
                                 args.batch_img, args.batch_txt, args.max_len, amp_dtype, k_list,
                                 args.save_csv_elip if args.save_csv_elip else None)
        print({f"ELIP R@{k}": round(rec_e[k], 4) for k in k_list})
        print({"ELIP mAP": round(map_e, 4)})

if __name__ == "__main__":
    main()
