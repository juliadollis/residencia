# workspace/v1/utils_image_cache.py
import os
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageFetcher:
    def __init__(self, cache_dir: str, timeout: int = 15, max_retries: int = 2):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries

    def fetch_pil(self, path: str):
        # aqui usamos apenas caminhos locais já salvos
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Imagem não encontrada: {path}")
        for _ in range(self.max_retries + 1):
            try:
                img = Image.open(path)
                img.load()
                return img.convert("RGB")
            except Exception:
                continue
        # última tentativa em modo permissivo
        try:
            img = Image.open(path)
            return img.convert("RGB")
        except Exception as e:
            raise e
