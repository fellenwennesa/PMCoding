from flask import Flask, request, jsonify, render_template
import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI  # Menggunakan langchain-openai

# Setup OpenAI API key
import os
from dotenv import load_dotenv

load_dotenv()  # Memuat variabel lingkungan dari file .env
openai.api_key = os.getenv("OPENAI_API_KEY")  # Menggunakan API key dari variabel lingkungan

app = Flask(__name__)

# Initialize LangChain with OpenAI
llm = OpenAI(openai_api_key=openai.api_key, temperature=0.7)  # Menyediakan API key secara eksplisit

# Create a template for the social media caption
caption_template = "Create a catchy social media caption about {topic}."
prompt = PromptTemplate(input_variables=["topic"], template=caption_template)
chain = LLMChain(llm=llm, prompt=prompt)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    data = request.json
    topic = data.get('topic')
    
    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    # Generate caption using LangChain
    caption = chain.run(topic)
    return jsonify({"caption": caption})

if __name__ == '__main__':
    app.run(debug=True)
