# /workspace/scripts/00_make_flickr30k_expanded.py
import os, argparse, random
from typing import List, Dict, Any
from datasets import load_dataset, DatasetDict
from tqdm import tqdm

SEED = 3407

def explode_batch(batch: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    Recebe um batch com:
      batch["image"]   -> lista de imagens (Image Feature ou dict com path/bytes)
      batch["caption"] -> lista de listas (5 captions por imagem)
    Retorna listas alinhadas (uma linha por caption).
    """
    images_out, texts_out = [], []
    for img, caps in zip(batch["image"], batch["caption"]):
        if isinstance(caps, list):
            for c in caps:
                if isinstance(c, str) and c.strip():
                    images_out.append(img)
                    texts_out.append(c.strip())
        else:
            images_out.append(img)
            texts_out.append(str(caps))
    return {"image": images_out, "text": texts_out}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--test_size", type=int, default=5000)
    args = ap.parse_args()

    random.seed(SEED)
    print(">> Carregando nlphuji/flickr30k (name=TEST)…")
    ds_raw = load_dataset("nlphuji/flickr30k", name="TEST", trust_remote_code=True)
    base_split = "test" if "test" in ds_raw else list(ds_raw.keys())[0]
    ds = ds_raw[base_split]  # ~31k imagens, cada uma com 5 captions

    print(">> Explodindo 5 captions por imagem (~155k pares)…")
    exploded = ds.map(
        explode_batch,
        batched=True,
        remove_columns=ds.column_names,
        desc="Explode captions",
    )

    total = len(exploded)
    print("Total de pares:", total)

    print(">> Embaralhando e criando holdout…")
    exploded = exploded.shuffle(seed=SEED)
    # por segurança, no máx. 10% para holdout
    test_size = min(args.test_size, max(1, total // 10))
    split = exploded.train_test_split(test_size=test_size, seed=SEED, shuffle=False)
    dsd = DatasetDict({"train": split["train"], "test": split["test"]})

    os.makedirs(args.out_dir, exist_ok=True)
    print(">> Salvando em disco:", args.out_dir)
    dsd.save_to_disk(args.out_dir)
    print("✓ Dataset expandido salvo.")

if __name__ == "__main__":
    main()