# download_html_corpus.py
import requests
import os
from bs4 import BeautifulSoup
import re


def download_html_corpus(output_dir="html_corpus", num_pages=1000):
    """Download HTML pages and convert to text"""
    os.makedirs(output_dir, exist_ok=True)

    # List of websites to crawl (diverse sources)
    seed_urls = [
        "https://en.wikipedia.org/wiki/Computer_science",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Mathematics",
        "https://en.wikipedia.org/wiki/Physics",
        "https://en.wikipedia.org/wiki/Biology",
        "https://www.bbc.com/news",
        "https://www.theguardian.com/international",
        "https://stackoverflow.com/questions",
        "https://github.com/explore",
    ]

    # Get more URLs from Wikipedia random pages
    for i in range(20):
        seed_urls.append(f"https://en.wikipedia.org/wiki/Special:Random")

    downloaded = 0
    for url in seed_urls:
        if downloaded >= num_pages:
            break

        try:
            print(f"Downloading {url}...")
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()

            # Get text content
            text = soup.get_text()

            # Clean text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            if len(text) > 500:  # Only save substantial content
                output_path = os.path.join(output_dir, f"html_{downloaded:06d}.txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                downloaded += 1
                print(f"Downloaded {downloaded}/{num_pages}")

        except Exception as e:
            print(f"Error downloading {url}: {e}")

    print(f"Downloaded {downloaded} HTML pages to {output_dir}")


if __name__ == "__main__":
    download_html_corpus("html_corpus", 1000)  # Download 1000 pages