# train_model.py
from src import CharLanguageModel, DataLoader
import argparse
import os

def train_model(corpus_type, corpus_path, model_save_path, download_if_missing=False):
    """Train language model for specific corpus type"""
    model = CharLanguageModel(n=7)

    # Download data if needed
    if download_if_missing and corpus_type == "email" and not os.path.exists(corpus_path):
        print("Downloading and preprocessing email corpus...")
        corpus_path = DataLoader.download_and_preprocess_enron(corpus_path)

    if corpus_type == "html":
        texts = DataLoader.load_html_corpus(corpus_path)
    elif corpus_type == "email":
        texts = DataLoader.load_email_corpus(corpus_path)
    elif corpus_type == "text":
        texts = DataLoader.load_text_files(corpus_path)
    else:
        raise ValueError(f"Unknown corpus type: {corpus_type}")

    print(f"Training on {len(texts)} documents...")

    for i, text in enumerate(texts):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(texts)} documents")
        model.train(text)

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_type", required=True, choices=["html", "email", "text"])
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--download", action="store_true", help="Download dataset if missing")

    args = parser.parse_args()
    train_model(args.corpus_type, args.corpus_path, args.model_path, args.download)