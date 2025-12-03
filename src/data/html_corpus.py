from pathlib import Path
from bs4 import BeautifulSoup
from .utils import stream_download
import logging

logger = logging.getLogger(__name__)

class HTMLCorpusDownloader:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)


    def save_html_as_text(self, url: str, index: int) -> bool:
        try:
            target = self.out_dir / f"html_{index:06d}.txt"
            r = stream_download(url, self.out_dir / f"tmp_{index}.html")
            html = r.read_bytes() if hasattr(r, "read_bytes") else open(r, "rb").read()
            soup = BeautifulSoup(html, "html.parser")
            for junk in soup(["script", "style", "nav", "header", "footer", "aside"]):
                junk.decompose()
            # text = soup.get_text(separator=" ")
            text = soup.get_text(separator=" ") # type: ignore[arg-type]
            # small cleaning
            text = " ".join(s.strip() for s in text.split())
            if len(text) > 500:
                target.write_text(text, encoding="utf-8")
                return True
            return False
        except Exception as e:
            logger.debug("Failed to fetch %s: %s", url, e)
            return False