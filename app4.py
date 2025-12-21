import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import re
import nltk
import os
from dotenv import load_dotenv
from typing import List, Optional, Union

# Import SimplerLLM dependencies
try:
    from SimplerLLM.language.llm import LLM, LLMProvider
except ImportError:
    print("Warning: SimplerLLM not found. Install it: pip install SimplerLLM")
    LLM = None
    LLMProvider = None

# --- Module Setup: Ensure NLTK resources are available ---
try:
    nltk.data.find('tokenizers/punkt')
except (nltk.downloader.DownloadError, LookupError):
    print("Downloading NLTK 'punkt' resource...")
    nltk.download('punkt', quiet=True)
    print("Download complete.")


class AITitleGenerator:
    """
    Extracts text from a URL and generates SEO titles using Groq's FREE Llama 3.
    """

    # URL/Scraping Constants
    DEFAULT_TIMEOUT: int = 30
    MIN_CONTENT_LENGTH: int = 100
    DEFAULT_HEADERS: dict = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }

    # LLM Constants
    LLM_PROMPT_TEMPLATE: str = """
I want you to act as a professional blog titles generator.
Think of titles that are SEO optimized and attention-grabbing at the same time.
Generate {num_titles} titles maximum in a numbered list format.
---
My blog post is about:
{content}
---
"""
    MAX_LLM_INPUT_LENGTH: int = 8000

    def __init__(self, groq_api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        if LLM is None:
            raise RuntimeError("SimplerLLM is not installed.")

        # FIX: Set environment variables that SimplerLLM uses internally for OpenAI-compatible providers
        os.environ["OPENAI_API_KEY"] = groq_api_key
        os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"

        print(f"🧠 Initializing Groq Engine: {model_name}...")

        # Now LLM.create will use the environment variables we just set
        self.llm_instance = LLM.create(
            provider=LLMProvider.OPENAI,
            model_name=model_name
        )

        self.session = self._setup_retry_session()
        print("✅ Groq Session initialized.")

    def _setup_retry_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def _extract_text_from_url(self, url: str) -> str:
        print(f"🔍 Fetching content from: {url}")
        try:
            response = self.session.get(url, headers=self.DEFAULT_HEADERS, timeout=self.DEFAULT_TIMEOUT)
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch URL: {e}")

        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()

        # Simple text extraction logic
        text_parts = [t.get_text(strip=True) for t in soup.find_all(['p', 'h1', 'h2', 'h3'])]
        text = " ".join(text_parts)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < self.MIN_CONTENT_LENGTH:
            raise ValueError("Extracted content is too short.")

        print(f"✅ Extracted {len(text)} characters.")
        return text

    def generate_titles_from_url(self, url: str, num_titles: int = 10) -> str:
        content = self._extract_text_from_url(url)
        safe_content = content[:self.MAX_LLM_INPUT_LENGTH]
        final_prompt = self.LLM_PROMPT_TEMPLATE.format(num_titles=num_titles, content=safe_content)

        print(f"✍️ Sending to Groq...")
        return self.llm_instance.generate_response(prompt=final_prompt)


# --- EXECUTION ---
if __name__ == "__main__":
    # Load variables from .env
    load_dotenv()

    # 1. GET YOUR FREE KEY: https://console.groq.com/keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    blog_url = "https://datasciencedojo.com/blog/from-llms-to-slms-in-agentic-ai/"

    if not GROQ_API_KEY:
        print("❌ Error: GROQ_API_KEY not found in .env file")
    else:
        try:
            generator = AITitleGenerator(groq_api_key=GROQ_API_KEY)
            titles = generator.generate_titles_from_url(url=blog_url, num_titles=10)
            print("\n🏆 Generated Titles:\n", titles)
        except Exception as e:
            print(f"\n❌ Error: {e}")