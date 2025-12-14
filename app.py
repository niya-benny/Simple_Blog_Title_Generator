import requests
from bs4 import BeautifulSoup
import re
import nltk
from transformers import pipeline

# Download punkt only if not already done, but usually it's cleaner to handle this outside of functions
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')


# -------- TEXT EXTRACTION FROM URL --------
# def extract_text_from_url(url):
#     # **FIX: Add a User-Agent header to mimic a web browser**
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#     }
#
#     try:
#         # Pass the headers to requests.get()
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()  # Raise HTTPError for bad status codes (4xx or 5xx)
#     except requests.exceptions.RequestException as e:
#         # Catch and re-raise to provide a clear error message
#         raise ValueError(f"Failed to fetch URL: {e}")
#
#     soup = BeautifulSoup(response.text, "html.parser")
#
#     # Remove scripts and styles
#     for tag in soup(["script", "style", "noscript"]):
#         tag.decompose()
#
#     # Optimized text extraction (using .stripped_strings is often better)
#     # Finding all <p> tags and joining their text.
#     text_parts = [p.get_text(separator=' ', strip=True) for p in soup.find_all("p")]
#     text = " ".join(text_parts)
#     text = re.sub(r"\s+", " ", text)  # Replace multiple spaces/newlines with a single space
#
#     return text.strip()

# -------- TEXT EXTRACTION FROM URL --------
def extract_text_from_url(url):
    # ADVANCED FIX: Include multiple headers to look like a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch URL: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts and styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text_parts = [p.get_text(separator=' ', strip=True) for p in soup.find_all("p")]
    text = " ".join(text_parts)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# -------- TITLE GENERATOR --------
def generate_blog_titles(content, num_titles=5):
    # Load the model outside the function if this function is called repeatedly for performance
    generator = pipeline(
        "text2text-generation",
        model="czearing/article-title-generator"
    )

    # Limit input length (important!)
    # The model 'czearing/article-title-generator' is a T5 model, which has a max input of 512 or 1024 tokens.
    # Limiting the content size is essential.
    content = content[:1000]

    titles = generator(
        content,
        max_length=20,
        num_return_sequences=num_titles,
        do_sample=True,
        temperature=0.9
    )

    return [t["generated_text"] for t in titles]


# -------- MAIN FUNCTION --------
def blog_title_generator(input_data, is_url=True):
    if is_url:
        print("🔍 Extracting blog content from URL...")
        content = extract_text_from_url(input_data)
    else:
        content = input_data

    if len(content) < 100:
        raise ValueError("Content too short to generate meaningful titles.")

    print("✍️ Generating blog titles...\n")
    titles = generate_blog_titles(content)

    return titles


# -------- EXAMPLE USAGE --------
if __name__ == "__main__":
    # OPTION 1: Blog URL
    url = "https://datasciencedojo.com/blog/from-llms-to-slms-in-agentic-ai/"

    try:
        # Your execution environment details
        print("Running Blog Title Generator...")

        # Run the generator
        titles = blog_title_generator(url, is_url=True)

        for i, title in enumerate(titles, 1):
            print(f"{i}. {title}")

    except ValueError as ve:
        print(f"An error occurred: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")