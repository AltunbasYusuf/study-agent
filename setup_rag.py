import chromadb
import ollama
from pypdf import PdfReader

PDF_PATH = "data/W06_Testing.pdf"
EMBED_MODEL = "bge-m3"
DB_PATH = "chroma_db"
COLLECTION_NAME = "study_material"

def get_embedding(text):
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]

def extract_chunks(pdf_path):
    reader = PdfReader(pdf_path)
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text().strip()
        if len(text) < 10:
            continue
        chunks.append({"text": text, "page": i + 1})
    return chunks

def main():
    print("Extracting chunks from PDF...")
    chunks = extract_chunks(PDF_PATH)
    print(f"Got {len(chunks)} chunks.\n")

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("Embedding and storing...")
    for chunk in chunks:
        embedding = get_embedding(chunk["text"])
        collection.add(
            ids=[f"page_{chunk['page']}"],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"page": chunk["page"]}],
        )
    print(f"Done. {collection.count()} chunks stored.")

if __name__ == "__main__":
    main()