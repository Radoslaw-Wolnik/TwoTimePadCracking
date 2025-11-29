import re
import email
from email import policy
from bs4 import BeautifulSoup

def preprocess_document(content, doc_type):
    """Convert document to byte stream based on type"""
    if doc_type == "email":
        # Extract plain text body
        msg = email.message_from_bytes(content)
        body = msg.get_body(('plain',)).get_content()
        return re.sub(r'\s+', ' ', body).encode('utf-8')

    elif doc_type == "html":
        # Strip HTML tags
        soup = BeautifulSoup(content, 'html.parser')
        return soup.get_text().encode('utf-8')

    elif doc_type == "word":
        # Extract text from .doc files (requires antiword)
        import subprocess
        result = subprocess.run(['antiword', '-'], input=content,
                                capture_output=True)
        return result.stdout

    elif doc_type == "text":
        # Basic text cleaning
        return re.sub(r'\s+', ' ', content.decode('utf-8', 'ignore')).encode('utf-8')

    return content  # Default: treat as bytes