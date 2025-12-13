import requests
from bs4 import BeautifulSoup
import re
import nltk
from transformers import pipeline

nltk.download('punkt')

# -------- TEXT EXTRACTION FROM URL --------
def extract_text_from_url(url):
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts and styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = " ".join(p.get_text() for p in soup.find_all("p"))
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -------- TITLE GENERATOR --------
def generate_blog_titles(content, num_titles=5):
    generator = pipeline(
        "text2text-generation",
        model="czearing/article-title-generator"
    )

    # Limit input length (important!)
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
    url = "https://example.com/blog-post"
    titles = blog_title_generator(url, is_url=True)

    # OPTION 2: Raw blog content
    # content = """Your blog text here..."""
    # titles = blog_title_generator(content, is_url=False)

    for i, title in enumerate(titles, 1):
        print(f"{i}. {title}")
