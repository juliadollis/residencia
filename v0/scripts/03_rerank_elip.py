# /workspace/scripts/03_rerank_elip.py
import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import yaml

# ====== mesmo mapper do treino ======
class ELIPMapper(nn.Module):
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
        out = self.net(t_cls)
        return out.view(-1, self.n_prompts, self.dim)


@dataclass
class CFG:
    baseline_out_dir: str   # /workspace/experiments/flickr30k_siglip_baseline
    elip_ckpt_dir: str      # /workspace/experiments/flickr30k_siglip_eliplite/elip_ckpts
    topk: int
    device: str


def load_cfg(path):
    with open(path, "r") as f:
        return CFG(**yaml.safe_load(f))


def compute_recall_at_k(sims: np.ndarray, gt, ks=[1,5,10]):
    order = np.argsort(-sims, axis=1)
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

    # baseline features
    xb = np.load(os.path.join(cfg.baseline_out_dir, "img_feats_test.npy")).astype(np.float32)  # [Nc, d]
    xq = np.load(os.path.join(cfg.baseline_out_dir, "text_feats_test.npy")).astype(np.float32) # [Nq, d]
    sims_base = xq @ xb.T
    n_q, d = xq.shape
    n_c = xb.shape[0]
    gt = [[i] for i in range(n_q)]

    # carregar melhor checkpoint
    ckpt = torch.load(os.path.join(cfg.elip_ckpt_dir, "best.pt"), map_location=cfg.device)
    dim = ckpt["dim"]; n_prompts = ckpt["n_prompts"]
    mapper = ELIPMapper(dim=dim, n_prompts=n_prompts).to(cfg.device)
    mapper.load_state_dict(ckpt["mapper"])
    projector = nn.Linear(dim, dim, bias=False).to(cfg.device)
    projector.load_state_dict(ckpt["projector"])
    mapper.eval(); projector.eval()

    # re-ranking top-k
    K = cfg.topk
    sims_new = sims_base.copy()
    xb_t = torch.from_numpy(xb).to(cfg.device)  # [Nc, d]

    with torch.no_grad():
        for i in range(n_q):
            # top-k candidatos pelo baseline
            topk_idx = np.argsort(-sims_base[i])[:K]
            v_base_k = xb_t[topk_idx]                 # [K, d]
            t_i = torch.from_numpy(xq[i:i+1]).to(cfg.device)  # [1, d]

            # prompts -> delta -> v_guided (apenas nos top-k)
            prompts = mapper(t_i.float())             # [1, n, d]
            delta = projector(prompts.mean(dim=1))    # [1, d]
            v_guided = nn.functional.normalize(v_base_k + delta, dim=-1)  # [K, d]
            t_norm = nn.functional.normalize(t_i, dim=-1)                  # [1, d]
            # novos scores só para os top-k
            new_scores = (t_norm @ v_guided.T).squeeze(0).cpu().numpy()    # [K]
            sims_new[i, topk_idx] = new_scores

    # métricas
    rec_base = compute_recall_at_k(sims_base, gt, ks=[1,5,10])
    rec_new  = compute_recall_at_k(sims_new,  gt, ks=[1,5,10])

    out = {
        "baseline": rec_base,
        "elip_lite": rec_new,
        "topk": K,
    }
    os.makedirs(cfg.elip_ckpt_dir, exist_ok=True)
    with open(os.path.join(cfg.elip_ckpt_dir, "rerank_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(">> Baseline Recall:", rec_base)
    print(">>   ELIP-lite Recall:", rec_new)
    print("✓ Re-ranking concluído. Métricas em", os.path.join(cfg.elip_ckpt_dir, "rerank_metrics.json"))


if __name__ == "__main__":
    main()