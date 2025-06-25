from __future__ import annotations

import json
import textwrap
from typing import List

import markdown
import pandas as pd
import requests
from chromadb import PersistentClient
from flask import Flask, jsonify, render_template, render_template_string, request

from embedding import SentenceTransformerEmbeddingFunction

# Config
OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral:7b"
OLLAMA_TIMEOUT = 120

CHROMA_PATH = "UroBot_database"
COLLECTION_NAME = "UroBot_v1.0"

# Flask & init
app = Flask(__name__)

embedding_func: SentenceTransformerEmbeddingFunction | None = None
collection = None

def initialise() -> None:
    global embedding_func, collection

    embedding_func = SentenceTransformerEmbeddingFunction()
    embedding_func.initialize_model()

    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
    )

# Helpers
def ollama_chat(model: str, messages: List[dict], temperature: float = 0.2) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
    }
    resp = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]

def convert_markdown_to_html_or_text(input_text: str) -> str:
    try:
        html = markdown.markdown(input_text, extensions=["tables"])
        return html
    except Exception:
        return f"<p>{input_text}</p>"

def process_query(question: str) -> tuple[str, list[str]]:
    assert collection is not None

    # 1. similarity search
    res = collection.query(query_texts=[question], n_results=9)

    context_blocks: list[str] = []
    documents_html: list[str] = []

    for idx, doc_txt in enumerate(res["documents"][0]):
        doc_id = res["ids"][0][idx]
        meta = res["metadatas"][0][idx]

        context_blocks.append(f"Document ID {doc_id[2:]}:\n{doc_txt}")

        if meta["paragraph_type"] == "table":

            df_html = pd.read_csv(meta["dataframe"]).to_html(index=False)
            documents_html.append(f"<h4>Document ID {doc_id[2:]}</h4>\n{df_html}")
        else:
            documents_html.append(
                f"<h4>Document ID {doc_id[2:]}</h4>\n{convert_markdown_to_html_or_text(doc_txt)}"
            )

    # 2. build system prompt
    context_text = "\n".join(context_blocks)

    system_prompt = textwrap.dedent(
        f"""
        És um assistente útil para responder a perguntas sobre o Regulamento Pedagógico da ESTG.
        Responde à pergunta do utilizador em linguagem simples e em português de Portugal.
        Usa as referências numeradas *Document ID* fornecidas entre as linhas --- como contexto.
        Cita cada facto que usares com o formato (Document ID N).
        Se o contexto não tiver a resposta, responde: "Desculpa, a minha base de conhecimento não inclui informação sobre esse tópico."
        ---
        {context_text}
        ---
        """
    ).strip()

    # 3. call Ollama
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    answer = ollama_chat(model=OLLAMA_MODEL, messages=messages, temperature=0.2)

    return answer, documents_html

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    documents = None
    query = ""

    if request.method == "POST":
        query = request.form["query"].strip()
        if query:
            answer, documents = process_query(query)

    return render_template(
        "index.html",
        answer=answer,
        query=query,
        documents=documents,
    )

if __name__ == "__main__":
    initialise()
    app.run(debug=True)
