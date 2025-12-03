import re

_RE_URL = re.compile(r"http\S+")
_RE_QUOTE = re.compile(r"^>.*$", flags=re.MULTILINE)
_RE_SIG = re.compile(r"^\s*--\s*$[\s\S]*", flags=re.MULTILINE)

def clean_email_body(text: str) -> str:
    # Normalize to str
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")

    # Remove content-type like headers that sometimes appear in bodies
    text = re.sub(r"^.*?Content-Type:.*?$", "", text, flags=re.IGNORECASE | re.DOTALL)


    text = _RE_URL.sub("", text)
    text = _RE_QUOTE.sub("", text)
    text = _RE_SIG.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()