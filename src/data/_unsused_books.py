import requests


def download_text_corpora():
    """Download existing text corpora that work well for this purpose"""
    corpora_sources = {
        "Gutenberg": "https://www.gutenberg.org/files/",  # Public domain books
        "WikiText": "https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/",
        "NewsCrawl": "http://data.statmt.org/news-crawl/",
    }

    # Example: Download Project Gutenberg books
    gutenberg_books = [
        "1342",  # Pride and Prejudice
        "84",  # Frankenstein
        "11",  # Alice in Wonderland
        "2701",  # Moby Dick
        # Add more book IDs
    ]

    for book_id in gutenberg_books:
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
        try:
            response = requests.get(url)
            # Save response.content to file
        except:
            pass