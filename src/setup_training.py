# setup_training_data.py
import os
from src.data.data_loader import DataLoader
from src.data.download_html_corpus import download_html_corpus


def setup_all_training_data():
    """Setup all training datasets"""

    # Setup email corpus
    if not os.path.exists("processed_emails"):
        print("Setting up email corpus...")
        DataLoader.download_and_preprocess_enron("processed_emails")

    # Setup HTML corpus
    if not os.path.exists("html_corpus"):
        print("Setting up HTML corpus...")
        download_html_corpus("html_corpus", 5000)  # Download 5000 pages

    # Setup text corpus (you can use any text files)
    if not os.path.exists("text_corpus"):
        print("Creating text corpus...")
        os.makedirs("text_corpus", exist_ok=True)
        # You can add your own text files here
        print("Add .txt files to text_corpus directory")

    print("Training data setup complete!")


if __name__ == "__main__":
    setup_all_training_data()