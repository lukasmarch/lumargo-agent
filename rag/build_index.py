# rag/build_index.py
import os, csv, glob
from pathlib import Path
from typing import List, Dict
from chromadb import Client
from chromadb import PersistentClient

# from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

# Wczytaj .env tylko lokalnie (na Render zmienne przyjdą z systemu)
load_dotenv()

# ── Stałe i ścieżki ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "data" / "chroma_db"
GALLERY_CSV = ROOT / "data" / "gallery.csv"
KB_DIR = ROOT / "data" / "kb"

GALLERY_COLL = "gallery"
KB_COLL = "knowledge_base"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Brakuje OPENAI_API_KEY. Ustaw w .env lub w ENV.")

client = OpenAI(api_key=OPENAI_API_KEY)


# ── Helpers ─────────────────────────────────────────────────────
def embed_texts(texts: List[str], model="text-embedding-3-small"):
    """Czyści wejście i zwraca listę wektorów."""
    if isinstance(texts, str):
        texts = [texts]
    cleaned: List[str] = []
    for t in texts:
        if isinstance(t, str):
            s = t.strip()
            if s:
                cleaned.append(s)
    if not cleaned:
        raise ValueError("embed_texts: empty input after cleaning")
    resp = client.embeddings.create(model=model, input=cleaned)
    return [d.embedding for d in resp.data]


def chunk(text: str, max_chars: int = 900) -> List[str]:
    """Dzieli dłuższy tekst na ~900 znaków, preferując podział po pustych liniach."""
    parts: List[str] = []
    buf = ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) + 2 <= max_chars:
            buf = (buf + "\n\n" + para) if buf else para
        else:
            if buf:
                parts.append(buf)
            if len(para) <= max_chars:
                buf = para
            else:
                for i in range(0, len(para), max_chars):
                    parts.append(para[i : i + max_chars])
                buf = ""
    if buf:
        parts.append(buf)
    return parts


# ── Budowa indeksu GALLERY ──────────────────────────────────────
def build_gallery() -> int:
    if not GALLERY_CSV.exists():
        print(f"[RAG] Brak pliku: {GALLERY_CSV}")
        return 0

    rows: List[Dict] = []
    with GALLERY_CSV.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # strip wszystkiego co stringowe
            clean = {
                k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()
            }
            rows.append(clean)

    print(f"[RAG] gallery.csv rows: {len(rows)}")
    if not rows:
        print("[RAG] gallery.csv is empty – skipping gallery index")
        return 0

    # Zmontuj dokumenty do embeddingów
    docs = []
    for clean in rows:
        doc = " | ".join(
            [
                clean.get("caption", "") or "",
                clean.get("tags", "") or "",
                f"typ:{clean.get('grave_type','') or ''}",
                f"kolor:{clean.get('plaque_color','') or ''}",
                f"kamień:{clean.get('stone','') or ''}",
                f"wykończenie:{clean.get('finish','') or ''}",
            ]
        ).strip(" |")
        if doc:
            docs.append(doc)

    print(f"[RAG] gallery non-empty docs: {len(docs)}")
    if not docs:
        print("[RAG] gallery: no non-empty docs assembled – sprawdź zawartość CSV")
        return 0

    chroma = PersistentClient(path=str(DB_DIR))
    # Zbuduj kolekcję od zera, bez lokalnego embeddera
    try:
        chroma.delete_collection(GALLERY_COLL)
    except Exception:
        pass
    coll = chroma.create_collection(GALLERY_COLL, embedding_function=None)

    # Oblicz embeddingi i zapisz
    embs = embed_texts(docs)
    n = len(embs)
    coll.add(
        ids=[row["id"] for row in rows][:n],
        documents=docs[:n],  # opcjonalne, ale użyteczne do debugowania
        embeddings=embs,
        metadatas=rows[:n],
    )

    # Log liczby po dodaniu
    try:
        cnt = coll.count()
        print(f"[RAG] gallery count(): {cnt}")
    except Exception:
        data = coll.get()  # bez include=["ids"]
        print(f"[RAG] gallery ids stored: {len((data.get('ids') or []))}")
    return n


# ── Budowa indeksu KB ───────────────────────────────────────────
def build_kb() -> int:
    files = sorted(KB_DIR.glob("*.*"))
    print(f"[RAG] kb files: {len(files)}")
    if not files:
        # Nie traktuj jako błąd — po prostu nie będzie QA-kontekstu
        return 0

    chroma = PersistentClient(path=str(DB_DIR))
    try:
        chroma.delete_collection(KB_COLL)
    except Exception:
        pass
    coll = chroma.create_collection(KB_COLL, embedding_function=None)

    ids, docs, metas = [], [], []
    for fp in files:
        txt = fp.read_text(encoding="utf-8").strip()
        if not txt:
            continue
        for i, ch in enumerate(chunk(txt)):
            s = (ch or "").strip()
            if not s:
                continue
            ids.append(f"{fp.name}#{i}")
            docs.append(s)
            metas.append({"source": fp.name, "chunk": i})

    print(f"[RAG] kb chunks to embed: {len(docs)}")
    if not docs:
        print("[RAG] kb: no documents – skipping")
        return 0

    embs = embed_texts(docs)
    coll.add(ids=ids, embeddings=embs, metadatas=metas)

    try:
        cnt = coll.count()
        print(f"[RAG] kb count(): {cnt}")
    except Exception:
        data = coll.get()
        print(f"[RAG] kb ids stored: {len((data.get('ids') or []))}")
    return len(embs)


# ── Orkiestracja ────────────────────────────────────────────────
def build_all():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[RAG] building DB in: {DB_DIR}")
    n_gal = build_gallery()
    n_kb = build_kb()
    print(f"[RAG] gallery: {n_gal} items, kb chunks: {n_kb}")


if __name__ == "__main__":
    build_all()
