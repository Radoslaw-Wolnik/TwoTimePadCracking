from pathlib import Path
from email import policy
import email
from .utils import safe_extract_tgz, stream_download, iter_text_files
from .preprocess import clean_email_body
import logging


logger = logging.getLogger(__name__)


ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20110402.tgz"

class EnronDownloader:
    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir


    def download(self, url: str = ENRON_URL, tgz_name: str = "enron_mail.tgz") -> Path:
        tgz_path = self.raw_dir / tgz_name
        if not tgz_path.exists():
            tgz_path = stream_download(url, tgz_path)
        else:
            logger.info("Using cached %s", tgz_path)
        return tgz_path


    def extract(self, tgz_path: Path, dest: Path) -> Path:
        logger.info("Extracting Enron to %s", dest)
        safe_extract_tgz(tgz_path, dest)
        return dest

class EnronPreprocessor:
    def __init__(self, source_dir: Path, output_dir: Path):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_email_body(self, path: Path) -> str | None:
        try:
            with open(path, "rb") as f:
                raw = f.read()

            msg = email.message_from_bytes(raw, policy=policy.default)

            # --- NEW API (EmailMessage) ---
            get_body = getattr(msg, "get_body", None)
            if callable(get_body):
                # Prefer plain text
                part = msg.get_body(preferencelist=("plain",))
                if part is None:
                    # fallback: any text/*
                    for p in msg.walk():
                        if p.get_content_maintype() == "text":
                            part = p
                            break
                if part is None:
                    return None

                try:
                    return part.get_content()
                except Exception:
                    pass  # fallback to legacy branch

            # --- OLD API fallback ---
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload is None:
                            continue
                        return payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    return payload.decode("utf-8", errors="ignore")

            return None

        except Exception as e:
            logger.debug("Failed to parse %s: %s", path, e)
            return None

    def process(self, min_length: int = 100, max_emails: int | None = None) -> int:
        count = 0
        written = 0

        for p in iter_text_files(self.source_dir, suffixes=("",)):
            # Stop if we've reached max_emails
            if max_emails is not None and written >= max_emails:
                logger.info(f"Reached maximum limit of {max_emails} emails")
                break

            body = self._get_email_body(p)
            if not body:
                continue

            cleaned = clean_email_body(body)
            if len(cleaned) >= min_length:
                out = self.output_dir / f"email_{written:06d}.txt"
                out.write_text(cleaned, encoding="utf-8")
                written += 1

            count += 1

            # Optional: Log progress every 1000 files processed
            if count % 1000 == 0:
                logger.info(f"Scanned {count} files, written {written} emails")

        logger.info(f"Processed {count} files, wrote {written} emails")
        if max_emails:
            logger.info(f"Limited to {max_emails} emails")
        return written