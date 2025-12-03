from pathlib import Path
from typing import Iterator
from .utils import iter_text_files
from .enron import EnronPreprocessor  # Add this import
import logging

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, res_dir: Path):
        self.res_dir = res_dir
        self.raw_dir = self.res_dir / "enron_emails_raw"
        self.processed_dir = self.res_dir / "processed_emails"
        self.models_dir = self.res_dir / "models"
        for d in (self.raw_dir, self.processed_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)

    def process_enron_emails(self, max_emails: int | None = None, min_length: int = 100) -> int:
        """Process Enron emails with optional limit."""
        preprocessor = EnronPreprocessor(self.raw_dir, self.processed_dir)
        return preprocessor.process(min_length=min_length, max_emails=max_emails)

    def iter_texts(self, directory: Path | str, recursive: bool = True) -> Iterator[str]:
        directory = Path(directory)
        for p in iter_text_files(directory, suffixes=(".txt",), recursive=recursive):
            try:
                yield p.read_text(encoding="utf-8")
            except Exception as e:
                logger.debug("Skipping %s: %s", p, e)