from pathlib import Path
import logging
from typing import Iterable
import requests
from tqdm import tqdm


logger = logging.getLogger(__name__)

def safe_extract_tgz(tgz_path: Path, dest: Path) -> None:
    import tarfile

    with tarfile.open(tgz_path, "r:gz") as tar:
        dest_real = dest.resolve()

        for member in tar.getmembers():
            member_path = dest / member.name
            member_real = member_path.resolve()

            # secure check
            if dest_real not in member_real.parents and member_real != dest_real:
                raise RuntimeError(f"Unsafe path in tarfile: {member.name}")

        tar.extractall(dest)

def stream_download(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return dest

def iter_text_files(root: Path, suffixes=(".txt",), recursive=True) -> Iterable[Path]:
    if recursive:
        for p in root.rglob("*"):
            if p.suffix.lower() in suffixes:
                yield p
    else:
        for p in root.iterdir():
            if p.suffix.lower() in suffixes:
                yield p