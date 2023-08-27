import os
from flask import Flask, send_from_directory, render_template, redirect, request, jsonify
from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain import PromptTemplate
import io
import requests

app = Flask(__name__)

port = int(os.environ.get("PORT", 5000))

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/<path:path>')
def all_routes(path):
    return redirect('/')


# Loading the model
def load_llm(max_tokens, prompt_template):
    # Load the locally downloaded model here
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=max_tokens,
        temperature=0.7
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )
    return llm_chain

# API route to generate article
@app.route('/generate_article', methods=['POST'])
def generate_article():
    data = request.json
    user_input = data.get('user_input')
    prompt_template = """You are a digital marketing and SEO expert and your task is to write an article so write an article on the given topic: {user_input}. The article must be under 800 words. The article should be be lengthy."""
    llm_call = load_llm(max_tokens=800, prompt_template=prompt_template)
    result = llm_call(user_input)
    if len(result) > 0:
        article = result['text']
        return jsonify({"article": article})
    else:
        return jsonify({"error": "Article couldn't be generated."}), 500

if __name__ == "__main__":
    app.run(port=port)
