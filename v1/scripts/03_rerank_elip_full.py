import os, json, argparse
from dataclasses import dataclass
import yaml, numpy as np, torch, torch.nn as nn
from datasets import load_dataset, load_from_disk
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

class ELIPMapper(nn.Module):
    def __init__(self, dim_in: int, dim_vision: int, n_prompts: int = 10):
        super().__init__()
        hid = dim_vision * n_prompts
        self.dim_in = dim_in; self.dim_vis = dim_vision; self.n_prompts = n_prompts
        self.net = nn.Sequential(
            nn.Linear(dim_in, hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, dim_vision * n_prompts),
        )
    def forward(self, t_feat: torch.Tensor) -> torch.Tensor:
        out = self.net(t_feat)
        return out.view(-1, self.n_prompts, self.dim_vis)

@dataclass
class CFG:
    baseline_out_dir: str
    elip_ckpt_dir: str
    dataset_id: str
    dataset_name: str
    image_column: str
    text_column: str
    caption_is_list: bool
    topk: int
    device: str

def load_cfg(path):
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    y["topk"]=int(y["topk"])
    return CFG(**y)

@torch.no_grad()
def get_text_proj(model, processor, texts, device):
    tx = processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(device)
    t = model.get_text_features(**tx)
    return nn.functional.normalize(t, dim=-1)

def forward_image_with_prompts(model, pixel_values, prompts, device):
    vision = model.vision_model
    pv = pixel_values.to(device)
    emb = vision.embeddings(pv)
    cls_tok, patch_tok = emb[:, :1, :], emb[:, 1:, :]
    x = torch.cat([cls_tok, prompts, patch_tok], dim=1)
    out = vision.encoder(inputs_embeds=x, output_hidden_states=False, return_dict=True)
    last = out.last_hidden_state
    pooled = vision.layernorm(last[:, 0, :])
    v = model.visual_projection(pooled)
    return nn.functional.normalize(v, dim=-1)

def recall_at_k(sims, ks=[1,5,10]):
    order = np.argsort(-sims, axis=1)
    N = order.shape[0]
    out={}
    for k in ks:
        hit=0
        for i in range(N):
            if i in set(order[i,:k].tolist()): hit+=1
        out[k]=hit/N
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True); args = ap.parse_args()
    cfg = load_cfg(args.cfg)

    # baseline feats (calculadas no holdout pré-treino)
    xb = np.load(os.path.join(cfg.baseline_out_dir,"img_feats_test.npy")).astype(np.float32)  # [Nc,d]
    xq = np.load(os.path.join(cfg.baseline_out_dir,"text_feats_test.npy")).astype(np.float32) # [Nq,d]
    with open(os.path.join(cfg.baseline_out_dir,"texts_test.json"),"r",encoding="utf-8") as f:
        raw_texts = json.load(f)
    sims_base = xq @ xb.T
    Nq, dproj = xq.shape
    rec_base = recall_at_k(sims_base)

    # carregamos o mesmo dataset do holdout (em disco se existir, senão remoto)
    if os.environ.get("LOCAL_DATASET_DIR"):
        dsd = load_from_disk(os.environ["LOCAL_DATASET_DIR"])
        ds_test = dsd["test"]
        image_column = cfg.image_column or "image"
    else:
        ds_raw = load_dataset(cfg.dataset_id, name=cfg.dataset_name, trust_remote_code=True)
        base_split="test" if "test" in ds_raw else list(ds_raw.keys())[0]
        raw = ds_raw[base_split]
        split = raw.train_test_split(test_size=1000, seed=3407, shuffle=True)
        ds_test = split["test"]; image_column = cfg.image_column

    # carregar checkpoint e modelo
    ckpt = torch.load(os.path.join(cfg.elip_ckpt_dir,"best.pt"), map_location=cfg.device)
    model_name = ckpt.get("model_name","google/siglip-base-patch16-224")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=torch.float16).to(cfg.device)
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]

    d_text_proj = ckpt["d_text_proj"]; d_vis = ckpt["d_vis"]; n_prompts = ckpt["n_prompts"]
    mapper = ELIPMapper(d_text_proj, d_vis, n_prompts).to(cfg.device)
    mapper.load_state_dict(ckpt["mapper"]); mapper.eval()
    alpha = ckpt.get("alpha", torch.tensor(1.0)).to(cfg.device)
    prompt_pos = ckpt.get("prompt_pos", torch.zeros((n_prompts, d_vis))).to(cfg.device)

    sims_new = sims_base.copy()
    K = int(cfg.topk)
    bs_img = 64

    for i in tqdm(range(Nq), desc="rerank"):
        idx = np.argsort(-sims_base[i])[:K]
        t_feat = get_text_proj(model, processor, [raw_texts[i]], cfg.device)      # [1,d_text_proj]
        p = alpha * (mapper(t_feat) + prompt_pos.unsqueeze(0))                     # [1,n,d_vis]

        new_scores=[]
        for s in range(0, len(idx), bs_img):
            sub = idx[s:s+bs_img]
            images = [ds_test[j][image_column] for j in sub]
            pix = processor(images=images, return_tensors="pt")["pixel_values"]
            v = forward_image_with_prompts(model, pix, p.expand(pix.size(0), -1, -1), cfg.device)  # [B,dproj]
            t_norm = torch.nn.functional.normalize(t_feat, dim=-1)
            sc = (t_norm @ v.T).squeeze(0).detach().cpu().numpy()
            new_scores.append(sc)
        sims_new[i, idx] = np.concatenate(new_scores, 0)

    rec_new = recall_at_k(sims_new)
    out = {"baseline": rec_base, "elip_full": rec_new, "topk": K}
    with open(os.path.join(cfg.elip_ckpt_dir,"rerank_full_metrics.json"),"w") as f:
        json.dump(out, f, indent=2)
    print(">> Baseline Recall:", rec_base)
    print(">>   ELIP-full Recall:", rec_new)
    print("✓ Re-ranking concluído. Métricas em", os.path.join(cfg.elip_ckpt_dir,"rerank_full_metrics.json"))

if __name__ == "__main__":
    main()