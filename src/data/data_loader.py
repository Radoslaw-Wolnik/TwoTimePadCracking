# src/data_loader.py
import os
import email
import re
import requests
import tarfile

class DataLoader:
    @staticmethod
    def load_email_corpus(corpus_path):
        """Load preprocessed email corpus"""
        texts = []
        for root, _, files in os.walk(corpus_path):
            for file in files:
                if file.endswith('.txt'):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'rb') as f:
                            content = f.read()
                            texts.append(content)
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
        return texts

    @staticmethod
    def load_html_corpus(corpus_path):
        """Load HTML corpus (preprocessed as text)"""
        return DataLoader.load_email_corpus(corpus_path)  # Same format

    @staticmethod
    def load_text_files(corpus_path):
        """Load plain text files"""
        return DataLoader.load_email_corpus(corpus_path)  # Same format

    @staticmethod
    def download_and_preprocess_enron(output_dir="processed_emails"):
        """Download and preprocess Enron emails in one step"""
        enron_raw_dir = "enron_emails_raw"

        # Download if not exists
        if not os.path.exists(enron_raw_dir):
            DataLoader._download_enron_emails(enron_raw_dir)

        # Preprocess
        DataLoader._preprocess_enron_emails(enron_raw_dir, output_dir)
        return output_dir

    @staticmethod
    def _download_enron_emails(output_dir):
        """Download Enron email dataset"""
        url = "https://www.cs.cmu.edu/~enron/enron_mail_20110402.tgz"
        local_path = "enron_emails.tgz"

        print("Downloading Enron dataset...")
        response = requests.get(url, stream=True)
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Extracting...")
        with tarfile.open(local_path, 'r:gz') as tar:
            tar.extractall(output_dir)

        os.remove(local_path)
        print(f"Enron emails saved to {output_dir}")

    @staticmethod
    def _preprocess_enron_emails(enron_dir, output_dir):
        """Extract and clean email bodies from Enron dataset"""
        os.makedirs(output_dir, exist_ok=True)

        email_count = 0
        for root, dirs, files in os.walk(enron_dir):
            for file in files:
                if file.endswith('.') or 'sent' in root.lower():  # Focus on sent emails
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            msg = email.message_from_file(f)

                            # Extract plain text body
                            body = None
                            if msg.is_multipart():
                                for part in msg.walk():
                                    if part.get_content_type() == "text/plain":
                                        body = part.get_payload(decode=True)
                                        break
                            else:
                                body = msg.get_payload(decode=True)

                            if body:
                                # Decode if needed
                                if isinstance(body, bytes):
                                    body = body.decode('utf-8', errors='ignore')

                                # Clean the email
                                cleaned = DataLoader._clean_email_body(body)

                                # Save cleaned email as bytes
                                if len(cleaned) > 100:  # Only save substantial emails
                                    output_path = os.path.join(output_dir, f"email_{email_count:06d}.txt")
                                    with open(output_path, 'wb') as out_f:  # Write as bytes
                                        out_f.write(cleaned.encode('utf-8'))
                                    email_count += 1

                    except Exception as e:
                        continue

        print(f"Processed {email_count} emails")

    @staticmethod
    def _clean_email_body(text):
        """Clean email text"""
        # Remove headers
        text = re.sub(r'^.*?Content-Type:.*?$', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove email signatures
        text = re.sub(r'^\s*--\s*$.*', '', text, flags=re.DOTALL | re.MULTILINE)

        # Remove quoted text
        text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()