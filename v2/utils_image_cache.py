import os
import time
import hashlib
import requests
from PIL import Image
from diskcache import Cache

def _hash(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()

class ImageFetcher:
    def __init__(self, cache_dir, timeout=15, max_retries=3, backoff=1.5):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.meta = Cache(os.path.join(cache_dir, "meta"))
        self.img_dir = os.path.join(cache_dir, "files")
        os.makedirs(self.img_dir, exist_ok=True)

    def _path_for(self, url):
        return os.path.join(self.img_dir, _hash(url) + ".jpg")

    def _is_valid_file(self, path):
        try:
            with Image.open(path) as im:
                im.verify()
            return True
        except:
            return False

    def _download(self, url, out_path):
        delay = 0.0
        for attempt in range(self.max_retries):
            if delay > 0:
                time.sleep(delay)
            try:
                r = requests.get(url, stream=True, timeout=self.timeout)
                ct = r.headers.get("Content-Type", "")
                if r.status_code == 200 and "image" in ct:
                    with open(out_path, "wb") as f:
                        for chunk in r.iter_content(8192):
                            if chunk:
                                f.write(chunk)
                    if self._is_valid_file(out_path):
                        self.meta[url] = {"ok": True, "ct": ct}
                        return out_path
                self.meta[url] = {"ok": False, "code": r.status_code, "ct": ct}
            except Exception as e:
                self.meta[url] = {"ok": False, "err": str(e)}
            delay = self.backoff if delay == 0 else delay * self.backoff
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except:
            pass
        return ""

    def fetch_pil(self, path_or_url):
        if hasattr(path_or_url, "convert"):
            return path_or_url.convert("RGB")
        if isinstance(path_or_url, str) and path_or_url.startswith("http"):
            out_path = self._path_for(path_or_url)
            if os.path.exists(out_path) and self._is_valid_file(out_path):
                return Image.open(out_path).convert("RGB")
            p = self._download(path_or_url, out_path)
            if p:
                return Image.open(p).convert("RGB")
            raise RuntimeError("falha ao baixar imagem")
        if isinstance(path_or_url, str):
            if not os.path.exists(path_or_url):
                raise RuntimeError("caminho inexistente")
            if not self._is_valid_file(path_or_url):
                raise RuntimeError("arquivo de imagem inválido")
            return Image.open(path_or_url).convert("RGB")
        raise RuntimeError("entrada de imagem inválida")
