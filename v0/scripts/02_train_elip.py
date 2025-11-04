# /workspace/scripts/02_train_elip.py
import os, json, argparse, math
from dataclasses import dataclass
from typing import Dict, List
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from transformers import AutoProcessor, AutoModel

# ======== Módulos ELIP-lite ========
class ELIPMapper(nn.Module):
    """
    Mapeia o embedding textual (dim d) -> n prompts (n, d).
    """
    def __init__(self, dim: int, n_prompts: int = 10):
        super().__init__()
        self.dim = dim
        self.n_prompts = n_prompts
        hid = dim * n_prompts
        self.net = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, hid),
        )

    def forward(self, t_cls: torch.Tensor) -> torch.Tensor:
        out = self.net(t_cls)  # [B, hid]
        return out.view(-1, self.n_prompts, self.dim)  # [B, n, d]


class PairwiseSigmoidLoss(nn.Module):
    """
    Sigmoid loss pareada (estilo SigLIP) sobre sim(t, v).
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        t = nn.functional.normalize(t, dim=-1)
        v = nn.functional.normalize(v, dim=-1)
        logits = (t @ v.T) / self.tau  # [B, B]
        # queremos a diagonal alta e o resto baixo
        pos = torch.diag(logits)  # [B]
        loss_pos = nn.functional.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
        mask = ~torch.eye(t.size(0), dtype=torch.bool, device=t.device)
        neg_logits = logits[mask]
        loss_neg = nn.functional.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        return loss_pos + loss_neg


# ======== Config ========
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

def load_cfg(path: str) -> CFG:
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    # defaults
    y.setdefault("n_prompts", 10)
    y.setdefault("lr", 1e-3)
    y.setdefault("epochs", 3)
    y.setdefault("temperature", 0.07)
    y.setdefault("save_every", 1)

    # casts defensivos (caso o YAML traga strings)
    y["batch_size"]   = int(y["batch_size"])
    y["num_workers"]  = int(y["num_workers"])
    y["n_prompts"]    = int(y["n_prompts"])
    y["epochs"]       = int(y["epochs"])
    y["save_every"]   = int(y["save_every"])
    y["lr"]           = float(y["lr"])
    y["temperature"]  = float(y["temperature"])

    return CFG(**y)


def seed_everything(seed=3407):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_images(batch, processor, image_column):
    images = [b[image_column] for b in batch]
    return processor(images=images, return_tensors="pt")


def collate_texts(batch, processor, text_column, caption_is_list: bool):
    def pick_text(x):
        v = x[text_column]
        if caption_is_list and isinstance(v, (list, tuple)) and len(v) > 0:
            return v[0]
        return v
    texts = [pick_text(b) for b in batch]
    return processor(text=texts, padding=True, truncation=True, return_tensors="pt"), texts


def build_dataloader(ds_split, cfg: CFG, processor):
    return DataLoader(
        ds_split,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        collate_fn=lambda b: (
            collate_images(b, processor, cfg.image_column),
            collate_texts(b, processor, cfg.text_column, cfg.caption_is_list)
        ),
        drop_last=True,  # importante p/ pairwise loss (B,B)
    )


@torch.no_grad()
def get_feats(model, pixel_inputs, text_inputs, device):
    # retorna (v_base [B,d], t_cls [B,d])
    pix = {k: v.to(device) for k, v in pixel_inputs.items()}
    txt = {k: v.to(device) for k, v in text_inputs.items()}
    v = model.get_image_features(**pix)    # [B, d]
    t = model.get_text_features(**txt)     # [B, d]
    return v, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)
    seed_everything(3407)
    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(cfg.out_dir, "elip_ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ----- dataset -----
    print(">> Carregando dataset:", cfg.dataset_id, "name=", cfg.dataset_name)
    ds_raw = load_dataset(cfg.dataset_id, name=cfg.dataset_name, trust_remote_code=True)
    base_split = "test" if "test" in ds_raw else list(ds_raw.keys())[0]
    raw = ds_raw[base_split]
    # mesmo split que no baseline: 1000 p/ eval, resto p/ treino
    splits = raw.train_test_split(test_size=1000, seed=3407, shuffle=True)
    ds_train, ds_val = splits["train"], splits["test"]

    # ----- modelo base (congelado) -----
    print(">> Carregando SigLIP:", cfg.model_name)
    processor = AutoProcessor.from_pretrained(cfg.model_name)
    base = AutoModel.from_pretrained(cfg.model_name, torch_dtype=torch.float16).to(cfg.device)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    # dimension
    with torch.no_grad():
        dummy = processor(text=["ok"], images=[ds_train[0][cfg.image_column]], return_tensors="pt")
        d_text = base.get_text_features(input_ids=dummy["input_ids"].to(cfg.device)).shape[-1]
        d_img  = base.get_image_features(pixel_values=dummy["pixel_values"].to(cfg.device)).shape[-1]
    assert d_text == d_img, "Esperado mesmo dim para texto e imagem"
    dim = d_text

    # ----- módulos treináveis -----
    mapper = ELIPMapper(dim=dim, n_prompts=cfg.n_prompts).to(cfg.device)
    # Pool das n prompts -> delta visual no espaço d
    pool = nn.AdaptiveAvgPool1d(1)  # vamos fazer mean ao longo de n
    projector = nn.Linear(dim, dim, bias=False).to(cfg.device)  # W

    optim = torch.optim.AdamW(list(mapper.parameters())+list(projector.parameters()), lr=cfg.lr)
    loss_fn = PairwiseSigmoidLoss(temperature=cfg.temperature)

    # ----- dataloaders -----
    train_loader = build_dataloader(ds_train, cfg, processor)
    val_loader   = build_dataloader(ds_val, cfg, processor)

    def guide_images(v_base, t_cls):
        # prompts: [B, n, d] -> mean over n -> [B,d] -> linear -> delta
        prompts = mapper(t_cls)                        # [B, n, d]
        pooled  = prompts.mean(dim=1)                  # [B, d]
        delta   = projector(pooled)                    # [B, d]
        v_guided = nn.functional.normalize(v_base + delta, dim=-1)
        return v_guided

    def evaluate():
        mapper.eval(); projector.eval()
        tot, tot_loss = 0, 0.0
        with torch.no_grad():
            for pix, (txt_in, _raw) in val_loader:
                v_base, t_cls = get_feats(base, pix, txt_in, cfg.device)
                v_base = v_base.float(); t_cls = t_cls.float()
                v_guided = guide_images(v_base, t_cls)
                loss = loss_fn(t_cls, v_guided)
                bs = t_cls.size(0)
                tot += bs; tot_loss += loss.item() * bs
        return tot_loss / max(1, tot)

    print(">> Treinando ELIP-lite (mapper + projector)…")
    global_step = 0
    best_val = float("inf")
    for epoch in range(1, cfg.epochs+1):
        mapper.train(); projector.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}")
        for pix, (txt_in, _raw) in pbar:
            v_base, t_cls = get_feats(base, pix, txt_in, cfg.device)
            v_base = v_base.float(); t_cls = t_cls.float()
            v_guided = guide_images(v_base, t_cls)
            loss = loss_fn(t_cls, v_guided)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(mapper.parameters())+list(projector.parameters()), max_norm=1.0)
            optim.step()

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # validação + checkpoint
        if (epoch % cfg.save_every) == 0:
            val_loss = evaluate()
            print(f"[val] epoch {epoch} loss={val_loss:.4f}")
            save_path = os.path.join(ckpt_dir, f"epoch{epoch}_val{val_loss:.4f}.pt")
            torch.save({
                "mapper": mapper.state_dict(),
                "projector": projector.state_dict(),
                "dim": dim,
                "n_prompts": cfg.n_prompts,
            }, save_path)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "mapper": mapper.state_dict(),
                    "projector": projector.state_dict(),
                    "dim": dim,
                    "n_prompts": cfg.n_prompts,
                }, os.path.join(ckpt_dir, "best.pt"))
                print("✓ best checkpoint atualizado")

    print("✓ Treino finalizado. Checkpoints em", ckpt_dir)


if __name__ == "__main__":
    main()