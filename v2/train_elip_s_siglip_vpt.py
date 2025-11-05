import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import math
import argparse
import torch
from datasets import load_from_disk
from transformers import AutoProcessor, AutoModel, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
from workspace.v1.utils_image_cache import ImageFetcher

class TextGuidedPromptMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_tokens, token_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, out_tokens * token_dim)
        self.out_tokens = out_tokens
        self.token_dim = token_dim
    def forward(self, txt_cls):
        x = self.fc1(txt_cls)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = x.view(x.size(0), self.out_tokens, self.token_dim)
        return x

class PairDS(torch.utils.data.Dataset):
    def __init__(self, ds, fetcher):
        self.ds = ds
        self.fetcher = fetcher
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        it = self.ds[i]
        return {"image_path": it["image_path"], "caption": it["caption"]}
    def fetch_images(self, paths):
        return [self.fetcher.fetch_pil(p) for p in paths]

def _collate(batch, proc, max_len, pairds):
    ims = pairds.fetch_images([b["image_path"] for b in batch])
    txt = [b["caption"] for b in batch]
    enc = proc(text=txt, images=ims, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return enc

def _pairwise_bce(pos, neg):
    logits = torch.cat([pos, neg], dim=0)
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=0)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

def _inject_vpt_and_encode(vision_model, pixel_values, prompt_tokens):
    target_dtype = next(vision_model.parameters()).dtype
    pixel_values = pixel_values.to(dtype=target_dtype)
    emb = vision_model.embeddings(pixel_values=pixel_values)
    x = emb[0] if isinstance(emb, tuple) else emb
    prompt_tokens = prompt_tokens.to(x.dtype)
    x = torch.cat([x, prompt_tokens], dim=1)
    enc = vision_model.encoder(inputs_embeds=x)
    h = enc[0]
    pooled = vision_model.post_layernorm(h[:, 0, :])
    return pooled

def _text_cls(text_model, input_ids, attention_mask):
    out = text_model(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    return out.last_hidden_state[:, 0, :]

def _try_enable_gc(module):
    ok = False
    if hasattr(module, "gradient_checkpointing_enable"):
        try:
            module.gradient_checkpointing_enable()
            ok = True
        except:
            pass
    if not ok and hasattr(module, "set_gradient_checkpointing"):
        try:
            module.set_gradient_checkpointing(True)
            ok = True
        except:
            pass
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="workspace/v1")
    ap.add_argument("--data_rel", type=str, default="data/COCOValMini")
    ap.add_argument("--model", type=str, default="google/siglip-base-patch16-224")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--per_device_batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--prompt_tokens", type=int, default=10)
    ap.add_argument("--hidden_dim", type=int, default=2048)
    ap.add_argument("--out_rel", type=str, default="ckpt_coco_valmini")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(args.root, args.data_rel)
    out_dir = os.path.join(args.root, args.out_rel)
    os.makedirs(out_dir, exist_ok=True)

    print(f">> Carregando dataset de {data_dir}")
    dsd = load_from_disk(data_dir)
    train_ds = dsd["train"]
    val_ds = dsd["validation"]
    print(f">> Tamanho train: {len(train_ds)}, val: {len(val_ds)}")

    print(">> Inicializando modelo base e processador...")
    fetcher = ImageFetcher(cache_dir=os.path.join(args.root, "cache", "images"), timeout=args.timeout, max_retries=args.retries)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if device.type == "cuda" else torch.float32)
    base = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype)
    base.to(device)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    if args.grad_ckpt:
        print(">> Ativando gradient checkpointing (se suportado)...")
        _ = _try_enable_gc(base)
        if hasattr(base, "vision_model"):
            _ = _try_enable_gc(base.vision_model)
            if hasattr(base.vision_model, "encoder"):
                _ = _try_enable_gc(base.vision_model.encoder)
        if hasattr(base, "text_model"):
            _ = _try_enable_gc(base.text_model)
            if hasattr(base.text_model, "encoder"):
                _ = _try_enable_gc(base.text_model.encoder)

    img_dim = base.config.vision_config.hidden_size
    txt_dim = base.config.text_config.hidden_size
    mlp = TextGuidedPromptMLP(txt_dim, args.hidden_dim, args.prompt_tokens, img_dim).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(mlp.parameters(), lr=args.lr)

    pair_train = PairDS(train_ds, fetcher)
    pair_val = PairDS(val_ds, fetcher)
    tr = DataLoader(pair_train, batch_size=args.per_device_batch, shuffle=True, num_workers=args.workers, collate_fn=lambda b: _collate(b, proc, args.max_len, pair_train))
    va = DataLoader(pair_val, batch_size=args.per_device_batch, shuffle=False, num_workers=args.workers, collate_fn=lambda b: _collate(b, proc, args.max_len, pair_val))

    steps_total = math.ceil(len(tr) / args.grad_accum) * args.epochs
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps, num_training_steps=steps_total)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.bf16))
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    best = 1e9

    print(">> Iniciando treinamento...\n")
    for ep in range(args.epochs):
        print(f"=== Época {ep+1}/{args.epochs} ===")
        mlp.train()
        opt.zero_grad(set_to_none=True)
        loop = tqdm(enumerate(tr), total=len(tr), desc=f"Treino Época {ep+1}")
        for i, batch in loop:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if "attention_mask" not in batch:
                batch["attention_mask"] = torch.ones_like(batch["input_ids"], device=device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
                tcls = _text_cls(base.text_model, batch["input_ids"], batch["attention_mask"])
                tcls = tcls.to(mlp.fc1.weight.dtype)
                prompts = mlp(tcls)
                vfeat = _inject_vpt_and_encode(base.vision_model, batch["pixel_values"], prompts)
                tfeat = base.get_text_features(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                vfeat = torch.nn.functional.normalize(vfeat, dim=-1)
                tfeat = torch.nn.functional.normalize(tfeat, dim=-1)
                pos = torch.sum(vfeat * tfeat, dim=-1)
                tperm = tfeat[torch.randperm(tfeat.size(0))]
                neg = torch.sum(vfeat * tperm, dim=-1)
                loss = _pairwise_bce(pos, neg) / args.grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (i + 1) % args.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                sch.step()
                opt.zero_grad(set_to_none=True)
            loop.set_postfix({"loss": float(loss.detach().cpu())})

        print(">> Avaliando no conjunto de validação...")
        mlp.eval()
        with torch.no_grad():
            vloss = 0.0
            vc = 0
            for batch in tqdm(va, desc="Validação"):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                if "attention_mask" not in batch:
                    batch["attention_mask"] = torch.ones_like(batch["input_ids"], device=device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
                    tcls = _text_cls(base.text_model, batch["input_ids"], batch["attention_mask"])
                    tcls = tcls.to(mlp.fc1.weight.dtype)
                    prompts = mlp(tcls)
                    vfeat = _inject_vpt_and_encode(base.vision_model, batch["pixel_values"], prompts)
                    tfeat = base.get_text_features(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    vfeat = torch.nn.functional.normalize(vfeat, dim=-1)
                    tfeat = torch.nn.functional.normalize(tfeat, dim=-1)
                    pos = torch.sum(vfeat * tfeat, dim=-1)
                    tperm = tfeat[torch.randperm(tfeat.size(0))]
                    neg = torch.sum(vfeat * tperm, dim=-1)
                    loss = _pairwise_bce(pos, neg)
                vloss += loss.item()
                vc += 1
            vloss /= max(1, vc)

        print(f">> Val Loss: {vloss:.6f}")
        last_p = os.path.join(out_dir, "last.pt")
        torch.save({"mlp": mlp.state_dict(), "cfg": vars(args), "base_model": args.model}, last_p)
        if vloss < best:
            best = vloss
            best_p = os.path.join(out_dir, "best.pt")
            torch.save({"mlp": mlp.state_dict(), "cfg": vars(args), "base_model": args.model}, best_p)
            print(f">> Novo melhor modelo salvo: {best_p} (val_loss={vloss:.6f})")

    print("\n>> Treinamento finalizado com sucesso!")

if __name__ == "__main__":
    main()
