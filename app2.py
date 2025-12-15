import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import re
import nltk
from transformers import pipeline, Pipeline
from typing import List, Optional, Union

# --- Module Setup: Ensure NLTK resources are available ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK 'punkt' resource...")
    nltk.download('punkt', quiet=True)
    print("Download complete.")


class URLTitleGenerator:
    """
    A specialized class for extracting contextually relevant text from a URL and
    generating accurate article titles using a transformer model.
    """
    # ... (rest of the class constants and __init__ remain the same) ...
    MODEL_NAME: str = "czearing/article-title-generator"
    MAX_MODEL_INPUT_LENGTH: int = 512
    DEFAULT_TIMEOUT: int = 30
    MIN_CONTENT_LENGTH: int = 100

    DEFAULT_HEADERS: dict = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Connection': 'keep-alive',
    }

    def __init__(self, model_name: Optional[str] = None):
        """Initializes the processor, sets up the retry session, and prepares for lazy-loading the model."""
        self._generator_pipeline: Optional[Pipeline] = None
        self.model_name = model_name if model_name else self.MODEL_NAME
        self.session = self._setup_retry_session()

    def _setup_retry_session(self) -> requests.Session:
        """Sets up a requests Session with a robust retry strategy."""
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=['HEAD', 'GET', 'OPTIONS'],
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    @property
    def generator_pipeline(self) -> Pipeline:
        """Lazy-loads the Hugging Face model pipeline."""
        if self._generator_pipeline is None:
            print(f"🧠 Loading NLP pipeline: {self.model_name}...")
            try:
                self._generator_pipeline = pipeline(
                    "text2text-generation",
                    model=self.model_name,
                    device=-1
                )
                # Ensure the model is loaded before proceeding
                if self._generator_pipeline:
                    print("✅ Pipeline loaded successfully.")
                else:
                    raise RuntimeError("Pipeline failed to initialize.")
            except Exception as e:
                raise RuntimeError(f"Failed to load Hugging Face model '{self.model_name}'. Error: {e}")
        return self._generator_pipeline

    def _extract_text_from_url(self, url: str) -> str:
        """
        Extracts clean, readable text from a given URL, prioritizing main article content.
        """
        print(f"🔍 Fetching content from: {url}")
        try:
            response = self.session.get(url, headers=self.DEFAULT_HEADERS, timeout=self.DEFAULT_TIMEOUT)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(
                f"Failed to fetch URL '{url}' after multiple retries: {type(e).__name__} - {e}")

        parser = 'lxml' if self._is_package_available('lxml') else 'html.parser'
        soup = BeautifulSoup(response.text, parser)

        # 1. Remove non-content elements
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()

        # 2. Advanced Heuristic for Main Content
        # Common selectors for main article content
        main_content_selectors = [
            'div[class*="article"]',
            'div[class*="post"]',
            'main',
            'article',
            'div[id*="content"]',
            'div[class*="content"]',
        ]

        main_content_block = None

        # Iterate over selectors to find the best (likely largest) block
        for selector in main_content_selectors:
            # Find the first element that matches the selector
            found_block = soup.select_one(selector)

            if found_block:
                # Prioritize a found block with a decent length
                text = found_block.get_text(separator=' ', strip=True)
                if len(text) > 500:  # Heuristic: if it's long, use it
                    main_content_block = found_block
                    print(f"   --> Found content block using selector: {selector}")
                    break  # Stop at the first robust match

        # Fallback: If no strong match is found, use the whole body
        if main_content_block is None:
            print("   --> Falling back to searching the entire body for text.")
            main_content_block = soup.body if soup.body else soup

        # 3. Extract and Clean Text
        # Now, extract text only from article-like elements within the identified block
        text_parts = []
        # Target main body text tags within the found block
        for tag in main_content_block.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li']):
            part = tag.get_text(separator=' ', strip=True)
            if part:
                text_parts.append(part)

        text = " ".join(text_parts)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < self.MIN_CONTENT_LENGTH:
            raise ValueError(
                f"Extracted content is too short (less than {self.MIN_CONTENT_LENGTH} characters) to be useful.")

        print(f"✅ Extracted {len(text)} characters of contextually relevant content.")
        return text

    # ... (rest of the methods: _generate_titles, generate_titles_from_url, _is_package_available remain the same) ...

    def _generate_titles(self, content: str, num_titles: int, **kwargs) -> List[str]:
        """Generates titles using the loaded transformer model."""
        safe_content = content[:self.MAX_MODEL_INPUT_LENGTH]
        print(f"✍️ Generating {num_titles} titles from content (truncated to {len(safe_content)} chars)...")

        titles_output = self.generator_pipeline(
            safe_content,
            num_beams=max(5, num_titles),
            num_return_sequences=num_titles,
            early_stopping=True,
            **kwargs
        )

        titles = [t["generated_text"].strip() for t in titles_output]
        return titles

    def generate_titles_from_url(
            self,
            url: str,
            num_titles: int = 5,
            **kwargs
    ) -> List[str]:
        """The main public method to process a URL and generate titles."""
        content = self._extract_text_from_url(url)
        return self._generate_titles(content, num_titles, **kwargs)

    @staticmethod
    def _is_package_available(package_name: str) -> bool:
        """Helper to check if an optional package is installed."""
        try:
            __import__(package_name)
            return True
        except ImportError:
            return False


# --- EXAMPLE USAGE (Main Function) ---
if __name__ == "__main__":
    blog_url = "https://datasciencedojo.com/blog/from-llms-to-slms-in-agentic-ai/"

    print("--- Running URL Article Title Generator ---")
    try:
        generator = URLTitleGenerator()
        titles = generator.generate_titles_from_url(url=blog_url, num_titles=5)

        print("\n🏆 Generated Titles:")
        for i, title in enumerate(titles, 1):
            print(f"{i}. **{title}**")

    except (ValueError, requests.exceptions.RequestException, RuntimeError) as e:
        print(f"\n❌ An operational error occurred: {e}")
    except Exception as e:
        print(f"\n❌ An UNEXPECTED error occurred: {e}")

