import os, json, argparse
from dataclasses import dataclass
from typing import Dict, Any
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import numpy as np
from transformers import AutoProcessor, AutoModel

# ---------- Mapper: texto-proj -> prompts em hidden_size do ViT ----------
class ELIPMapper(nn.Module):
    def __init__(self, dim_in: int, dim_vision: int, n_prompts: int = 10):
        super().__init__()
        self.dim_in = dim_in
        self.dim_vis = dim_vision
        self.n_prompts = n_prompts
        hid = dim_vision * n_prompts
        self.net = nn.Sequential(
            nn.Linear(dim_in, hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, dim_vision * n_prompts),
        )
    def forward(self, t_feat: torch.Tensor) -> torch.Tensor:
        out = self.net(t_feat)  # [B, n*d]
        return out.view(-1, self.n_prompts, self.dim_vis)

# ---------- Sigmoid pairwise loss ----------
class PairwiseSigmoidLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__(); self.tau = temperature
    def forward(self, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        t = nn.functional.normalize(t, dim=-1)
        v = nn.functional.normalize(v, dim=-1)
        logits = (t @ v.T) / self.tau
        pos = torch.diag(logits)
        loss_pos = nn.functional.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
        mask = ~torch.eye(t.size(0), dtype=torch.bool, device=t.device)
        loss_neg = nn.functional.binary_cross_entropy_with_logits(logits[mask], torch.zeros_like(logits[mask]))
        return loss_pos + loss_neg

# ---------- Config ----------
@dataclass
class CFG:
    dataset_id: str
    dataset_name: str
    image_column: str
    text_column: str
    caption_is_list: bool
    split_train: str
    split_test: str
    model_name: str
    batch_size: int
    num_workers: int
    device: str
    out_dir: str
    n_prompts: int
    lr: float
    epochs: int
    temperature: float
    save_every: int
    lambda_delta: float

def load_cfg(path: str) -> CFG:
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    y.setdefault("n_prompts", 10)
    y.setdefault("lr", 5e-4)
    y.setdefault("epochs", 5)
    y.setdefault("temperature", 0.07)
    y.setdefault("save_every", 1)
    y.setdefault("lambda_delta", 1e-3)
    # casts
    y["batch_size"]=int(y["batch_size"]); y["num_workers"]=int(y["num_workers"])
    y["n_prompts"]=int(y["n_prompts"]); y["epochs"]=int(y["epochs"])
    y["save_every"]=int(y["save_every"]); y["lr"]=float(y["lr"])
    y["temperature"]=float(y["temperature"]); y["lambda_delta"]=float(y["lambda_delta"])
    return CFG(**y)

def seed_everything(seed=3407):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def collate_images(batch, processor, image_column):
    images = [b[image_column] for b in batch]
    return processor(images=images, return_tensors="pt")

def collate_texts(batch, processor, text_column, caption_is_list: bool):
    def pick_text(x):
        v = x[text_column]
        if caption_is_list and isinstance(v, (list, tuple)) and len(v)>0: return v[0]
        return v
    texts = [pick_text(b) for b in batch]
    return processor(text=texts, padding=True, truncation=True, return_tensors="pt"), texts

def build_dataloader(ds_split, cfg: CFG, processor):
    return DataLoader(
        ds_split, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        shuffle=True, drop_last=True,
        collate_fn=lambda b: (collate_images(b, processor, cfg.image_column),
                              collate_texts(b, processor, cfg.text_column, cfg.caption_is_list))
    )

@torch.no_grad()
def get_text_proj(model: AutoModel, text_inputs: Dict[str, torch.Tensor], device: str):
    tx = {k:v.to(device) for k,v in text_inputs.items()}
    t = model.get_text_features(**tx)
    return t.float()

def forward_image_with_prompts(model: AutoModel, pixel_inputs: Dict[str, torch.Tensor],
                               prompts: torch.Tensor, device: str) -> torch.Tensor:
    vision = model.vision_model
    pix = {k:v.to(device) for k,v in pixel_inputs.items()}
    emb = vision.embeddings(pix["pixel_values"])   # [B,1+N,d_vis]
    cls_tok, patch_tok = emb[:, :1, :], emb[:, 1:, :]
    x = torch.cat([cls_tok, prompts, patch_tok], dim=1)
    out = vision.encoder(inputs_embeds=x, output_hidden_states=False, return_dict=True)
    last = out.last_hidden_state
    pooled = vision.layernorm(last[:, 0, :])
    v = model.visual_projection(pooled)
    return nn.functional.normalize(v, dim=-1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True); args = ap.parse_args()
    cfg = load_cfg(args.cfg); seed_everything(3407)
    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(cfg.out_dir, "elip_full_ckpts"); os.makedirs(ckpt_dir, exist_ok=True)

    # ----- dataset -----
    if os.environ.get("LOCAL_DATASET_DIR"):
        local_dir = os.environ["LOCAL_DATASET_DIR"]
        print(">> load_from_disk:", local_dir)
        dsd = load_from_disk(local_dir)
        ds_train, ds_val = dsd["train"], dsd["test"]
        # no dataset expandido, caption já está em "text"
        cfg.caption_is_list = False
        cfg.text_column = cfg.text_column or "text"
        cfg.image_column = cfg.image_column or "image"
    else:
        print(">> Carregando remoto:", cfg.dataset_id, "name=", cfg.dataset_name)
        ds_raw = load_dataset(cfg.dataset_id, name=cfg.dataset_name, trust_remote_code=True)
        base_split = "test" if "test" in ds_raw else list(ds_raw.keys())[0]
        raw = ds_raw[base_split]
        splits = raw.train_test_split(test_size=1000, seed=3407, shuffle=True)
        ds_train, ds_val = splits["train"], splits["test"]

    # ----- modelo base (congelado) -----
    print(">> Carregando SigLIP:", cfg.model_name)
    processor = AutoProcessor.from_pretrained(cfg.model_name)
    base = AutoModel.from_pretrained(cfg.model_name, dtype=torch.float16).to(cfg.device)
    base.eval()
    for p in base.parameters(): p.requires_grad_(False)

    # dims
    d_text_proj = base.get_text_features(**processor(text=["ok"], return_tensors="pt").to(cfg.device)).shape[-1]
    with torch.no_grad():
        tmp = processor(images=[ds_train[0][cfg.image_column]], return_tensors="pt").to(cfg.device)
        vtmp = base.vision_model.embeddings(tmp["pixel_values"])
    d_vis = vtmp.shape[-1]

    # módulos treináveis
    mapper = ELIPMapper(d_text_proj, d_vis, cfg.n_prompts).to(cfg.device)
    alpha = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=cfg.device))
    prompt_pos = torch.zeros((cfg.n_prompts, d_vis), device=cfg.device, dtype=torch.float32, requires_grad=True)

    params = list(mapper.parameters()) + [alpha, prompt_pos]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.01)
    loss_fn = PairwiseSigmoidLoss(cfg.temperature)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    train_loader = build_dataloader(ds_train, cfg, processor)
    val_loader   = build_dataloader(ds_val,   cfg, processor)

    def make_prompts(t_feat):
        p = mapper(t_feat)                          # [B,n,d_vis]
        p = p + prompt_pos.unsqueeze(0)             # pos embeddings opcionais
        p = alpha * p                               
        return p

    @torch.no_grad()
    def eval_loss():
        mapper.eval()
        tot, tot_loss = 0, 0.0
        for pix,(tx,_raw) in val_loader:
            t = get_text_proj(base, tx, cfg.device)
            p = make_prompts(t)
            v = forward_image_with_prompts(base, pix, p, cfg.device)
            loss = loss_fn(t, v)
            bs = t.size(0); tot += bs; tot_loss += loss.item()*bs
        return tot_loss/max(1, tot)

    print(">> Treinando ELIP-FULL…")
    best = float("inf")
    for epoch in range(1, cfg.epochs+1):
        mapper.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}")
        for pix,(tx,_raw) in pbar:
            t = get_text_proj(base, tx, cfg.device)         # [B,d_text_proj]
            p = make_prompts(t)                              # [B,n,d_vis]
            v = forward_image_with_prompts(base, pix, p, cfg.device)  # [B,d_text_proj]
            loss_main = loss_fn(t, v)
            reg = cfg.lambda_delta * (p.pow(2).mean())
            loss = loss_main + reg

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val = eval_loss()
        print(f"[val] epoch {epoch} loss={val:.4f}")
        if val < best:
            best = val
            torch.save({
                "mapper": mapper.state_dict(),
                "alpha": alpha.detach().cpu(),
                "prompt_pos": prompt_pos.detach().cpu(),
                "d_text_proj": d_text_proj,
                "d_vis": d_vis,
                "n_prompts": cfg.n_prompts,
                "model_name": cfg.model_name,
            }, os.path.join(ckpt_dir, "best.pt"))
            print("✓ best checkpoint atualizado")
        if (epoch % cfg.save_every)==0:
            torch.save({
                "mapper": mapper.state_dict(),
                "alpha": alpha.detach().cpu(),
                "prompt_pos": prompt_pos.detach().cpu(),
                "d_text_proj": d_text_proj,
                "d_vis": d_vis,
                "n_prompts": cfg.n_prompts,
                "model_name": cfg.model_name,
            }, os.path.join(ckpt_dir, f"epoch{epoch}.pt"))
        scheduler.step()

    print("✓ Treino finalizado. Checkpoints em", ckpt_dir)

if __name__ == "__main__":
    main()