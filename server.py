import os
from flask import Flask, send_from_directory, render_template, redirect, request, jsonify
from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings

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

NOMBRE_INDICE_CHROMA = "/db2"



embedding_instruct = HuggingFaceInstructEmbeddings(
    model_name="all-mpnet-base-v2",
    model_kwargs={"device":"cpu"}
)

vectorstore_chroma = Chroma(
    persist_directory=NOMBRE_INDICE_CHROMA,
    embedding_function=embedding_instruct
)

# Loading the model
def load_llm(max_tokens, prompt_template):
    # Load the locally downloaded model here
    docs_q = vectorstore_chroma.similarity_search_with_score(query, k=3)

    return llm_chain

# API route to generate article
@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST':
        # Obtener la consulta del formulario HTML
        query = request.form.get('query')

        # Realizar la búsqueda de similitud de texto en el índice Chroma
        docs_q = vectorstore_chroma.similarity_search_with_score(query, k=1)

        # Puedes procesar los resultados y enviarlos de vuelta al cliente como JSON, por ejemplo:
        result = {
            'query': query,
            'most_similar_doc': docs_q[0] if docs_q else None  # El documento más similar (si hay resultados)
        }

        return jsonify(result)

    # Manejar otras condiciones si es necesario

# ...

if __name__ == "__main__":
    app.run(port=port)
