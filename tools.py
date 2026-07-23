import os
import chromadb
import ollama
from langchain_core.tools import tool

EMBED_MODEL = "bge-m3"
CHAT_MODEL = "qwen2.5:7b"
DB_PATH = "chroma_db"
COLLECTION_NAME = "study_material"
SUMMARIES_DIR = "summaries"

os.makedirs(SUMMARIES_DIR, exist_ok=True)

_client = chromadb.PersistentClient(path=DB_PATH)
_collection = _client.get_collection(COLLECTION_NAME)


def _get_embedding(text: str):
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def _retrieve(query: str, n_results: int = 3):
    query_embedding = _get_embedding(query)
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "page": results["metadatas"][0][i]["page"],
            "text": results["documents"][0][i],
        })
    return chunks


def _build_context(chunks):
    return "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in chunks)


@tool
def ask_pdf(question: str) -> str:
    """Answers a specific question using the content of the study PDF (software testing lecture slides).
    Use this when the user asks something that could be answered directly from the course material,
    e.g. 'What is the difference between verification and validation?'"""

    chunks = _retrieve(question, n_results=3)
    context = _build_context(chunks)

    system_prompt = """You are a study assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say so clearly instead of guessing.
Always cite the page number(s) your answer is based on."""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0},
    )
    return response["message"]["content"]


@tool
def save_summary(topic: str) -> str:
    """Creates a study summary about a specific topic from the PDF and saves it as a Markdown file.
    Use this when the user asks for notes, a summary, or a study sheet on a topic,
    e.g. 'Make me a summary of testing strategies' or 'Save notes on black-box testing'."""

    chunks = _retrieve(topic, n_results=8)
    context = _build_context(chunks)

    system_prompt = """You are a study assistant. Write a clear, well-structured study summary
about the given topic, using ONLY the provided context. Use Markdown formatting
(headings, bullet points). Include page number references. If the context does not
cover the topic well, say so at the top of the summary instead of inventing content."""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nTopic: {topic}"},
        ],
        options={"temperature": 0},
    )
    summary_text = response["message"]["content"]

    normalized_topic = topic.lower().strip()
    safe_filename = "".join(c if c.isalnum() or c in " -_" else "" for c in normalized_topic).strip()
    safe_filename = safe_filename.replace(" ", "_")[:80] or "summary"
    filepath = os.path.join(SUMMARIES_DIR, f"{safe_filename}.md")
    filepath = os.path.join(SUMMARIES_DIR, f"{safe_filename}.md")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Summary: {topic}\n\n{summary_text}\n")

    return f"Summary saved to {filepath}"