# 📝 Simple Blog Title Generator

An AI-powered web application that extracts content from a blog URL and generates **SEO-optimized, attention-grabbing blog titles** using **Groq’s free LLaMA 3 models** via the **SimplerLLM** framework.

This project supports both:

* **Command-line execution**, and
* **Flask-based web interface** for interactive usage.

---

## 🚀 Features

* 🔗 Extracts meaningful textual content directly from a blog URL
* 🧹 Cleans HTML by removing scripts, styles, headers, footers, and navigation
* 🧠 Uses **LLaMA 3 (70B)** via Groq for high-quality title generation
* 📈 Generates **SEO-friendly, click-worthy blog titles**
* 🔁 Built-in retry mechanism for robust HTTP requests
* 🌐 Flask API for frontend or client integration
* ⚡ Fast inference using Groq’s free LLM API

---

## 🛠️ Tech Stack

* **Python 3.9+**
* **Flask** – Web framework
* **BeautifulSoup4** – Web scraping
* **Requests + Retry Adapter** – Reliable HTTP fetching
* **NLTK** – Text processing
* **SimplerLLM** – Unified LLM interface
* **Groq API** – LLaMA 3 inference
* **dotenv** – Secure environment variable management

---

## 🔑 Getting a Free Groq API Key

1. Visit 👉 [https://console.groq.com/keys](https://console.groq.com/keys)
2. Generate a **FREE API key**
3. Create a `.env` file in your project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/simple_blog_title_generator.git
cd simple_blog_title_generator

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---


You can modify the blog URL and number of titles inside the script:

```python
blog_url = "https://example.com/blog-post"
titles = generator.generate_titles_from_url(url=blog_url, num_titles=10)
```

---




## ⚠️ Error Handling

* Invalid or unreachable URLs
* Pages with insufficient content
* Missing API key
* LLM initialization failures

All errors are returned gracefully with meaningful messages.

---

## 🧠 How It Works (High Level)

1. Fetches webpage content using `requests`
2. Cleans and extracts meaningful text using `BeautifulSoup`
3. Truncates input safely to fit LLM limits
4. Sends structured prompt to Groq-hosted LLaMA 3
5. Returns numbered, SEO-optimized titles

---

## 📌 Limitations

* Requires public URLs (no paywalled content)
* Content-heavy pages work best
* Free Groq tier has request limits

---

## 🔮 Future Improvements

* ✨ Keyword-based title generation
* 📊 SEO score per title
* 🌍 Multi-language support
* 🧩 Chrome extension integration
* 🗂️ Title history & export

---

## 👨‍💻 Author

Developed by **Niya Benny**
*Data Science Student | AI & ML Enthusiast*

---


