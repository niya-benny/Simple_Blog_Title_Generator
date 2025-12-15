import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import re
import nltk
from typing import List, Optional, Union

# Import your SimplerLLM dependencies (assuming these are installed and configured)
try:
    # NOTE: You must have SimplerLLM and an OpenAI API key configured for this to run.
    from SimplerLLM.language.llm import LLM, LLMProvider 
except ImportError:
    print("Warning: SimplerLLM not found. Install it to use the GPT-4o feature.")
    LLM = None
    LLMProvider = None


# --- Module Setup: Ensure NLTK resources are available ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK 'punkt' resource...")
    nltk.download('punkt', quiet=True)
    print("Download complete.")


class AITitleGenerator:
    """
    A universal class for extracting text from a URL and generating high-quality
    SEO-optimized titles using a powerful LLM (e.g., GPT-4o).
    """

    # URL/Scraping Constants
    DEFAULT_TIMEOUT: int = 30 
    MIN_CONTENT_LENGTH: int = 100
    # Common headers for robust web scraping
    DEFAULT_HEADERS: dict = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Connection': 'keep-alive',
    }

    # LLM Constants
    LLM_PROMPT_TEMPLATE: str = """
I want you to act as a professional blog titles generator.
Think of titles that are SEO optimized and attention-grabbing at the same time,
and will encourage people to click and read the blog post.
Generate {num_titles} titles maximum in a numbered list format.
---
My blog post is about the following content:
{content}
---
"""
    # LLMs handle larger contexts, but we limit for speed and cost.
    MAX_LLM_INPUT_LENGTH: int = 8000 

    def __init__(self, provider: LLMProvider = LLMProvider.OPENAI, model_name: str = "gpt-4o"):
        """Initializes the processor and the LLM instance."""
        if LLM is None:
            raise RuntimeError("SimplerLLM must be installed and configured with API keys.")
            
        print(f"🧠 Initializing LLM: {model_name} from {provider.name}...")
        self.llm_instance = LLM.create(provider=provider, model_name=model_name)
        self.session = self._setup_retry_session()
        print("✅ LLM and Session initialized.")

    # --- Web Scraping Methods (Copied for robustness and universality) ---

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
    
    # Static helper method remains the same
    @staticmethod
    def _is_package_available(package_name: str) -> bool:
        """Helper to check if an optional package is installed."""
        try:
            __import__(package_name)
            return True
        except ImportError:
            return False

    def _extract_text_from_url(self, url: str) -> str:
        """
        Extracts clean, readable text from a given URL, prioritizing main article content.
        (Uses the robust logic developed previously)
        """
        print(f"🔍 Fetching content from: {url}")
        try:
            response = self.session.get(url, headers=self.DEFAULT_HEADERS, timeout=self.DEFAULT_TIMEOUT)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Failed to fetch URL '{url}' after multiple retries: {type(e).__name__} - {e}")

        parser = 'lxml' if self._is_package_available('lxml') else 'html.parser'
        soup = BeautifulSoup(response.text, parser)

        # 1. Remove non-content elements
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()

        # 2. Advanced Heuristic for Main Content
        main_content_selectors = [
            'div[class*="article"]', 
            'div[class*="post"]',
            'main', 
            'article', 
            'div[id*="content"]', 
            'div[class*="content"]',
        ]
        
        main_content_block = None
        for selector in main_content_selectors:
            found_block = soup.select_one(selector)
            if found_block:
                text = found_block.get_text(separator=' ', strip=True)
                if len(text) > 500: 
                    main_content_block = found_block
                    print(f"   --> Found content block using selector: {selector}")
                    break
        
        if main_content_block is None:
            print("   --> Falling back to searching the entire body for text.")
            main_content_block = soup.body if soup.body else soup

        # 3. Extract and Clean Text
        text_parts = []
        for tag in main_content_block.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li']): 
            part = tag.get_text(separator=' ', strip=True)
            if part:
                text_parts.append(part)

        text = " ".join(text_parts)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < self.MIN_CONTENT_LENGTH:
             raise ValueError(f"Extracted content is too short (less than {self.MIN_CONTENT_LENGTH} characters) to be useful.")
        
        print(f"✅ Extracted {len(text)} characters of contextually relevant content.")
        return text

    # --- LLM Generation Method ---

    def generate_titles_from_url(
        self, 
        url: str, 
        num_titles: int = 10,
    ) -> str:
        """
        The main public method to process a URL and generate titles using the LLM.

        Returns:
            The raw text response from the LLM (which should be a numbered list).
        """
        # 1. Extract Content
        content = self._extract_text_from_url(url)
        
        # 2. Prepare Prompt
        # Truncate content to fit within the LLM's context window (and manage cost/speed)
        safe_content = content[:self.MAX_LLM_INPUT_LENGTH] 
        
        final_prompt = self.LLM_PROMPT_TEMPLATE.format(
            num_titles=num_titles,
            content=safe_content
        )
        
        print(f"✍️ Sending {len(safe_content)} characters of content to GPT-4o to generate {num_titles} titles...")

        # 3. Generate Response
        generated_text = self.llm_instance.generate_response(prompt=final_prompt)
        
        return generated_text

# --- EXAMPLE USAGE (Main Function) ---
if __name__ == "__main__":
    # Use the same URL as before for consistency
    blog_url = "https://datasciencedojo.com/blog/from-llms-to-slms-in-agentic-ai/"

    print("--- Running AI (GPT-4o) Article Title Generator ---")
    try:
        # NOTE: This line requires SimplerLLM and your OpenAI API key to be configured.
        generator = AITitleGenerator(provider=LLMProvider.OPENAI, model_name="gpt-4o")
        
        # Generate 10 titles
        titles_output = generator.generate_titles_from_url(url=blog_url, num_titles=10)

        print("\n🏆 Generated Titles (from GPT-4o):")
        print(titles_output)

    except (ValueError, requests.exceptions.RequestException, RuntimeError) as e:
        print(f"\n❌ An operational error occurred: {e}")
    except Exception as e:
        print(f"\n❌ An UNEXPECTED error occurred: {e}")
