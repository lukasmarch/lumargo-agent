import os
from typing import List, Dict
from chromadb import PersistentClient

# from chromadb.config import Settings
from openai import OpenAI

from dotenv import load_dotenv
from rag.embeddings import embed_query

load_dotenv()  # wczyta plik .env z katalogu projektu, jeśli istnieje

DB_DIR = "data/chroma_db"
KB_COLL = "knowledge_base"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def retrieve(question: str, k: int = 5):
    chroma = PersistentClient(path=DB_DIR)
    coll = chroma.get_collection(KB_COLL)
    qv = embed_query(question)
    res = coll.query(query_embeddings=[qv], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return [
        {"text": d, "source": m.get("source"), "chunk": m.get("chunk")}
        for d, m in zip(docs, metas)
    ]


def answer_kb(question: str) -> Dict:
    ctx = retrieve(question, k=5)
    context_text = "\n\n---\n\n".join(
        [f"[{c['source']}#{c['chunk']}]\n{c['text']}" for c in ctx]
    )
    system = (
        "Jesteś doradcą Lumargo. Odpowiadaj wyłącznie na podstawie dostarczonego kontekstu. "
        "Jeśli brakuje danych, powiedz to wprost i zaproponuj pomiar/kontakt. "
        "Na końcu wypisz listę źródeł w formacie: [plik#chunk]."
    )
    msgs = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Pytanie: {question}\n\nKontekst:\n{context_text}",
        },
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=msgs, temperature=0.2
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": [f"{c['source']}#{c['chunk']}" for c in ctx],
    }
