# workspace/v1/train_elip_s_siglip_vpt.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from tqdm import tqdm
from functools import partial
from transformers import AutoProcessor, AutoModel, get_linear_schedule_with_warmup
from collections import defaultdict

from workspace.v1.utils_image_cache import ImageFetcher

# estabilidade em dataloader no CUDA + multiprocess
torch.multiprocessing.set_start_method("spawn", force=True)

# sementes determinísticas básicas
GLOBAL_SEED = 3407
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED); torch.manual_seed(GLOBAL_SEED)


# -------------------------
# Dataset simples de pares
# -------------------------
class PairDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        ex = self.ds[i]
        return {"image_path": ex["image_path"], "caption": ex["caption"]}


# -----------------------------------------
# MLP que gera tokens de prompt guiado por texto
# -----------------------------------------
class TextGuidedPromptMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_tokens, token_dim, init_std=0.01, dtype=torch.float32):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim, dtype=dtype)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self.fc3 = torch.nn.Linear(hidden_dim, out_tokens * token_dim, dtype=dtype)
        self.act = torch.nn.GELU()
        self.out_tokens = out_tokens
        self.token_dim = token_dim
        with torch.no_grad():
            for m in [self.fc1, self.fc2, self.fc3]:
                torch.nn.init.trunc_normal_(m.weight, std=init_std)
                torch.nn.init.zeros_(m.bias)

    def forward(self, txt_cls):
        x = self.act(self.fc1(txt_cls))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x.view(x.size(0), self.out_tokens, self.token_dim)


# -----------------------------------------
# Extrai CLS textual da cabeça do encoder
# -----------------------------------------
def get_text_cls(text_model, input_ids, attention_mask):
    out = text_model(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    return out.last_hidden_state[:, 0, :]


# ---------------------------------------------------
# Injeta os tokens de prompt após o CLS visual (SigLIP)
# ---------------------------------------------------
def inject_after_cls(vision_model, pixel_values, prompt_tokens):
    """
    pixel_values: [B, 3, H, W]  (no device do modelo)
    prompt_tokens: [B, P, D]    (no mesmo device do modelo)
    retorna: vfeat [B, D]
    """
    target_dtype = next(vision_model.parameters()).dtype
    x = vision_model.embeddings(pixel_values=pixel_values.to(dtype=target_dtype))
    x = x[0] if isinstance(x, tuple) else x  # [B, 1+N, D]
    prompt_tokens = prompt_tokens.to(x.dtype)

    cls, rest = x[:, :1, :], x[:, 1:, :]
    # concatena [CLS] + [PROMPTS] + [PATCHES]
    x = torch.cat([cls, prompt_tokens, rest], dim=1)

    h = vision_model.encoder(inputs_embeds=x)[0]  # [B, 1+P+N, D]
    v = vision_model.post_layernorm(h[:, 0, :])   # CLS final
    return v


# ---------------------------------------------------
# Collates
#   A) _collate_unique: 1 imagem por batch (default)
#   B) _collate_multi: múltiplas captions da mesma imagem no batch
# OBS IMPORTANTE: NÃO mover para device no collate (para permitir pin_memory)
# ---------------------------------------------------
def _collate_unique(batch, proc, max_len, fetcher):
    # agrupa por image_path e escolhe UMA caption por imagem (evita falsos negativos)
    by_img = defaultdict(list)
    for ex in batch:
        by_img[ex["image_path"]].append(ex["caption"])

    uniq_paths, captions = [], []
    for p, caps in by_img.items():
        uniq_paths.append(p)
        captions.append(random.choice(caps))

    # abre imagens (descarta corrompidas)
    pils, good_paths, good_caps = [], [], []
    for p, c in zip(uniq_paths, captions):
        try:
            pils.append(fetcher.fetch_pil(p))
            good_paths.append(p); good_caps.append(c)
        except Exception:
            continue

    enc_im = proc(images=pils, return_tensors="pt")   # CPU tensors
    enc_tx = proc(text=good_caps, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    if "attention_mask" not in enc_tx:
        enc_tx["attention_mask"] = torch.ones_like(enc_tx["input_ids"])

    img_ids = torch.arange(len(good_paths), dtype=torch.long)  # CPU
    return {
        "pixel_values": enc_im["pixel_values"],   # CPU
        "input_ids": enc_tx["input_ids"],         # CPU
        "attention_mask": enc_tx["attention_mask"],  # CPU
        "image_ids": img_ids,                     # CPU
        "image_paths": good_paths,
        "captions": good_caps,
    }


def _collate_multi(batch, proc, max_len, fetcher):
    # permite várias captions da mesma imagem no mesmo batch (p/ multi-positive InfoNCE)
    paths, caps = [], []
    for ex in batch:
        paths.append(ex["image_path"]); caps.append(ex["caption"])

    pils, keep_idx = [], []
    for i, p in enumerate(paths):
        try:
            pils.append(fetcher.fetch_pil(p)); keep_idx.append(i)
        except Exception:
            continue
    if len(keep_idx) == 0:
        # fallback para unique se toda a leva falhar
        return _collate_unique(batch, proc, max_len, fetcher)

    paths = [paths[i] for i in keep_idx]
    caps  = [caps[i]  for i in keep_idx]

    enc_im = proc(images=pils, return_tensors="pt")   # CPU
    enc_tx = proc(text=caps, padding=True, truncation=True, max_length=max_len, return_tensors="pt")  # CPU
    if "attention_mask" not in enc_tx:
        enc_tx["attention_mask"] = torch.ones_like(enc_tx["input_ids"])

    # image_ids: mesmo id para amostras que compartilham a mesma imagem
    uniq = {}
    ids = []
    for p in paths:
        if p not in uniq: uniq[p] = len(uniq)
        ids.append(uniq[p])
    img_ids = torch.tensor(ids, dtype=torch.long)  # CPU

    return {
        "pixel_values": enc_im["pixel_values"],
        "input_ids": enc_tx["input_ids"],
        "attention_mask": enc_tx["attention_mask"],
        "image_ids": img_ids,
        "image_paths": paths,
        "captions": caps,
    }


# -------------------------
# Losses
# -------------------------
def info_nce(sim, tau=0.07):
    logits = sim / tau
    target = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(logits, target)

def multi_positive_infonce(sim, img_ids, tau=0.07):
    logits = sim / tau  # [B, B]
    pos_mask = (img_ids.unsqueeze(1) == img_ids.unsqueeze(0))  # [B,B], bool
    pos_logits = torch.where(pos_mask, logits, torch.full_like(logits, float("-inf")))
    pos_logsumexp = torch.logsumexp(pos_logits, dim=1)  # [B]
    all_logsumexp = torch.logsumexp(logits, dim=1)      # [B]
    return -(pos_logsumexp - all_logsumexp).mean()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="workspace/v1")
    ap.add_argument("--data_rel", type=str, default="data/WikiArtDesc")
    ap.add_argument("--model", type=str, default="google/siglip-base-patch16-224")

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--per_device_batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_steps", type=int, default=1000)

    ap.add_argument("--prompt_tokens", type=int, default=10)
    ap.add_argument("--hidden_dim", type=int, default=2048)
    ap.add_argument("--out_rel", type=str, default="ckpt_wikiart_v2")

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=0)  # 0 = mais estável em container
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--multi_positive", action="store_true",
                    help="usa InfoNCE com múltiplos positivos no batch (deixe sem para '1 imagem por batch')")

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # -------------------------
    # Dataset
    # -------------------------
    data_dir = os.path.join(args.root, args.data_rel)
    print(f">> Carregando dataset de {data_dir}")
    dsd = load_from_disk(data_dir)
    pair_train = PairDataset(dsd["train"])
    pair_val   = PairDataset(dsd["validation"])
    print(f">> Tamanho train: {len(pair_train)}, val: {len(pair_val)}")

    # -------------------------
    # Modelo base + Processor
    # -------------------------
    print(">> Inicializando modelo base e processador...")
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if device.type == "cuda" else torch.float32)
    base = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype).to(device)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    if args.grad_ckpt and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()

    img_dim = base.config.vision_config.hidden_size
    txt_dim = base.config.text_config.hidden_size
    mlp = TextGuidedPromptMLP(txt_dim, args.hidden_dim, args.prompt_tokens, img_dim, dtype=dtype).to(device)

    opt = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=0.0)
    total_steps = max(1, (len(pair_train) // max(1, args.per_device_batch)) * args.epochs // max(1, args.grad_accum))
    sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=min(args.warmup_steps, total_steps), num_training_steps=total_steps)

    fetcher = ImageFetcher(cache_dir=os.path.join(args.root, "cache", "images"))

    collate_fn = partial(_collate_multi if args.multi_positive else _collate_unique,
                         proc=proc, max_len=args.max_len, fetcher=fetcher)

    tr = DataLoader(
        pair_train,
        batch_size=args.per_device_batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    va = DataLoader(
        pair_val,
        batch_size=max(1, args.per_device_batch * 2),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    # AMP API nova
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda" and not args.bf16))

    # -------------------------
    # Treinamento
    # -------------------------
    out_dir = os.path.join(args.root, args.out_rel)
    os.makedirs(out_dir, exist_ok=True)
    best_loss = float("inf")

    print(">> Iniciando treinamento...")
    for ep in range(1, args.epochs + 1):
        mlp.train()
        loop = tqdm(tr, desc=f"Treino Época {ep}", total=len(tr))
        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(loop):
            # === mover para device AQUI (vêm do collate em CPU) ===
            pixel_values  = batch["pixel_values"].to(device, non_blocking=True)
            input_ids     = batch["input_ids"].to(device, non_blocking=True)
            attention_mask= batch["attention_mask"].to(device, non_blocking=True)
            image_ids     = batch["image_ids"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda"), dtype=amp_dtype):
                # texto -> CLS
                tcls = get_text_cls(base.text_model, input_ids, attention_mask)
                # gera prompts [B, P, D]
                prompts = mlp(tcls.to(next(mlp.parameters()).dtype))
                # imagem -> feat com injeção (ATENÇÃO: NÃO repete os prompts no treino!)
                vfeat = inject_after_cls(base.vision_model, pixel_values, prompts)
                vfeat = F.normalize(vfeat, dim=-1)
                # texto -> embedding
                tfeat = F.normalize(base.get_text_features(input_ids=input_ids,
                                                           attention_mask=attention_mask), dim=-1)
                # similaridade
                sim = tfeat @ vfeat.T  # [B,B]

                if args.multi_positive:
                    loss = multi_positive_infonce(sim, image_ids, tau=args.tau)
                else:
                    loss = info_nce(sim, tau=args.tau)

            scaler.scale(loss / max(1, args.grad_accum)).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sch.step()

            loop.set_postfix(loss=float(loss.detach().cpu().item()))

        # -------------------------
        # Validação (loss média)
        # -------------------------
        mlp.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(va, desc="Validação"):
                pixel_values  = batch["pixel_values"].to(device, non_blocking=True)
                input_ids     = batch["input_ids"].to(device, non_blocking=True)
                attention_mask= batch["attention_mask"].to(device, non_blocking=True)
                image_ids     = batch["image_ids"].to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=(device.type == "cuda"), dtype=amp_dtype):
                    tcls = get_text_cls(base.text_model, input_ids, attention_mask)
                    prompts = mlp(tcls.to(next(mlp.parameters()).dtype))
                    vfeat = inject_after_cls(base.vision_model, pixel_values, prompts)  # sem repeat no val
                    vfeat = F.normalize(vfeat, dim=-1)
                    tfeat = F.normalize(base.get_text_features(input_ids=input_ids,
                                                               attention_mask=attention_mask), dim=-1)
                    sim = tfeat @ vfeat.T
                    if args.multi_positive:
                        loss = multi_positive_infonce(sim, image_ids, tau=args.tau)
                    else:
                        loss = info_nce(sim, tau=args.tau)
                val_losses.append(float(loss.detach().cpu().item()))
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        # salva ckpts
        state = {
            "mlp": mlp.state_dict(),
            "cfg": {"hidden_dim": args.hidden_dim, "prompt_tokens": args.prompt_tokens},
        }
        torch.save(state, os.path.join(out_dir, "last.pt"))
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(state, os.path.join(out_dir, "best.pt"))

        print(f">> Época {ep} concluída | val_loss={val_loss:.4f} | best={best_loss:.4f}")


if __name__ == "__main__":
    main()
