import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Import your existing logic
# (Assuming the AITitleGenerator class code provided in your prompt is here)
# For brevity, I am showing the Flask integration below.

load_dotenv()

app = Flask(__name__)

# Initialize the generator globally for efficiency
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
generator = None

if GROQ_API_KEY:
    try:
        # Assuming the class from your prompt is defined above this
        generator = AITitleGenerator(groq_api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"Initialization Error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if not generator:
        return jsonify({"error": "LLM Generator not initialized. Check API Key."}), 500
    
    data = request.json
    url = data.get('url')
    num_titles = data.get('num_titles', 5)

    if not url:
        return jsonify({"error": "URL is required"}), 400

    try:
        titles_raw = generator.generate_titles_from_url(url=url, num_titles=num_titles)
        # Convert the numbered list string into a Python list for better UI rendering
        titles_list = [t.strip() for t in titles_raw.strip().split('\n') if t.strip()]
        return jsonify({"titles": titles_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
