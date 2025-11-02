# rag/build_index.py
import os, csv, glob, json
from pathlib import Path
from typing import List, Dict, Any
from chromadb import Client
from chromadb import PersistentClient

# from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

# from embeddings
from rag.embeddings import embed_texts

# Wczytaj .env tylko lokalnie (na Render zmienne przyjdą z systemu)
load_dotenv()

# ── Stałe i ścieżki ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "data" / "chroma_db"
GALLERY_CSV = ROOT / "data" / "gallery.csv"
KB_DIR = ROOT / "data" / "kb"

GALLERY_COLL = "gallery"
KB_COLL = "knowledge_base"

HEADSTONE_FIELD_MAP = {
    "style": "headstone_style",
    "letter_material": "headstone_letter_material",
    "letter_technique": "headstone_letter_technique",
    "has_photo_on_headboard": "headstone_has_photo_on_headboard",
    "photo_type": "headstone_photo_type",
    "headboard_shape": "headstone_headboard_shape",
    "cover_type": "headstone_cover_type",
    "accessories": "headstone_accessories",
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Brakuje OPENAI_API_KEY. Ustaw w .env lub w ENV.")

client = OpenAI(api_key=OPENAI_API_KEY)


# ── Helpers ─────────────────────────────────────────────────────
# def embed_texts(texts: List[str], model="text-embedding-3-small"):
#     """Czyści wejście i zwraca listę wektorów."""
#     if isinstance(texts, str):
#         texts = [texts]
#     cleaned: List[str] = []
#     for t in texts:
#         if isinstance(t, str):
#             s = t.strip()
#             if s:
#                 cleaned.append(s)
#     if not cleaned:
#         raise ValueError("embed_texts: empty input after cleaning")
#     resp = client.embeddings.create(model=model, input=cleaned)
#     return [d.embedding for d in resp.data]


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


def _parse_headstone(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    if not text or text.lower() == "nan":
        return {}
    try:
        data = json.loads(text)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _clean_value(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        return text if text else "nan"
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(items) if items else "nan"
    if value is None:
        return "nan"
    return str(value)


def _fmt_pair(label: str, value: Any) -> str:
    text = _clean_value(value)
    if text.lower() == "nan":
        return ""
    return f"{label}:{text}"


# ── Budowa indeksu GALLERY ──────────────────────────────────────
def build_gallery() -> int:
    if not GALLERY_CSV.exists():
        print(f"[RAG] Brak pliku: {GALLERY_CSV}")
        return 0

    rows: List[Dict] = []
    with GALLERY_CSV.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            clean: Dict[str, Any] = {}
            for k, v in row.items():
                if not isinstance(k, str) or not k.strip():
                    continue  # pomiń kolumny bez nagłówka (None lub "")
                key = k.strip()
                clean[key] = v.strip() if isinstance(v, str) else v
            headstone_meta = _parse_headstone(clean.get("headstone"))
            if headstone_meta:
                clean["headstone"] = json.dumps(
                    headstone_meta, ensure_ascii=False, separators=(",", ":")
                )
            else:
                clean["headstone"] = _clean_value(clean.get("headstone"))

            for target in HEADSTONE_FIELD_MAP.values():
                clean[target] = "nan"

            clean["grave_type"] = _clean_value(clean.get("grave_type"))

            if headstone_meta:
                for source, target in HEADSTONE_FIELD_MAP.items():
                    clean[target] = _clean_value(headstone_meta.get(source))

                grave_val = _clean_value(headstone_meta.get("grave_type"))
                if grave_val.lower() != "nan":
                    clean["grave_type"] = grave_val

            rows.append(clean)

    print(f"[RAG] gallery.csv rows: {len(rows)}")
    if not rows:
        print("[RAG] gallery.csv is empty – skipping gallery index")
        return 0

    # Zmontuj dokumenty do embeddingów
    docs = []
    for clean in rows:
        caption = _clean_value(clean.get("caption"))
        tags = _clean_value(clean.get("tags"))
        parts = []
        if caption.lower() != "nan":
            parts.append(caption)
        if tags.lower() != "nan":
            parts.append(tags)
        parts.extend(
            filter(
                None,
                [
                    _fmt_pair("typ", clean.get("grave_type")),
                    _fmt_pair("kolor", clean.get("plaque_color")),
                    _fmt_pair("kamień", clean.get("stone")),
                    _fmt_pair("wykończenie", clean.get("finish")),
                    _fmt_pair("styl", clean.get("headstone_style")),
                    _fmt_pair("litery", clean.get("headstone_letter_material")),
                    _fmt_pair("technika", clean.get("headstone_letter_technique")),
                    _fmt_pair("foto", clean.get("headstone_has_photo_on_headboard")),
                    _fmt_pair("zdjecie", clean.get("headstone_photo_type")),
                    _fmt_pair("kształt", clean.get("headstone_headboard_shape")),
                    _fmt_pair("pokrycie", clean.get("headstone_cover_type")),
                    _fmt_pair("akcesoria", clean.get("headstone_accessories")),
                ],
            )
        )
        doc = " | ".join(parts).strip(" |")
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
    coll.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

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
