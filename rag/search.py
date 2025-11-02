# rag/search.py
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from chromadb import Client

from chromadb import PersistentClient

# from chromadb.config import Settings
from openai import OpenAI

# from embeddings
from rag.embeddings import embed_texts

# Ścieżki niezależne od bieżącego katalogu
ROOT = Path(__file__).resolve().parents[1]
DB_DIR = str(ROOT / "data" / "chroma_db")

KB_COLL = "knowledge_base"

GALLERY_COLL = "gallery"
ALLOWED_FILTERS = {
    "grave_type",
    "plaque_color",
    "stone",
    "finish",
    "category",
    "headstone_has_photo_on_headboard",
    "headstone_photo_type",
    "style",
}

FILTER_ALIASES = {
    "has_photo_on_headboard": "headstone_has_photo_on_headboard",
    "photo_type": "headstone_photo_type",
}


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# def embed(q: str):
#     return (
#         client.embeddings.create(model="text-embedding-3-small", input=[q])
#         .data[0]
#         .embedding
#     )
def embed(q: str):
    return embed_texts(q)[0]


def build_where(filters: Optional[Dict[str, Any]]):
    """
    Buduje filtr zgodny z nowszym API Chroma:
    - pojedynczy warunek: {"field": {"$eq":"value"}}
    - wiele warunków: {"$and":[{"field":{"$eq":"v1"}},{"field2":{"$eq":"v2"}}]}
    """
    if not filters:
        return None
    clauses = []
    for k, v in filters.items():
        key = FILTER_ALIASES.get(k, k)
        if isinstance(v, str) and v.strip().lower() == "nan":
            continue
        if key in ALLOWED_FILTERS and v not in (None, "", []):
            clauses.append({key: {"$eq": v}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def search_gallery(
    query: str, filters: Optional[Dict[str, Any]] = None, limit: int = 3
) -> List[Dict[str, Any]]:
    chroma = PersistentClient(path=DB_DIR)
    try:
        coll = chroma.get_collection(GALLERY_COLL)

    except Exception:
        # Kolekcja nie istnieje – zwróć pustą listę zamiast 500
        return []

    where = build_where(filters)
    try:
        qv = embed(query or "")
        res = coll.query(query_embeddings=[qv], n_results=max(limit, 1), where=where)
    except Exception:
        # Błąd zapytania (np. zły where) -> zwracamy pustą listę
        return []

    metas = res.get("metadatas", [[]])[0] or []
    ids = res.get("ids", [[]])[0] or []

    out: List[Dict[str, Any]] = []
    for i, meta in enumerate(metas):
        if not meta:
            continue
        out.append(
            {
                "id": ids[i] if i < len(ids) else "",
                "image_url": meta.get("image_url", ""),
                "thumb_url": meta.get("thumb_url") or meta.get("image_url", ""),
                "caption": meta.get("caption", ""),
                "grave_type": meta.get("grave_type"),
                "plaque_color": meta.get("plaque_color"),
                "stone": meta.get("stone"),
                "finish": meta.get("finish"),
                "headstone": meta.get("headstone", "nan"),
                "headstone_style": meta.get("headstone_style"),
                "headstone_letter_material": meta.get("headstone_letter_material"),
                "headstone_letter_technique": meta.get("headstone_letter_technique"),
                "headstone_has_photo_on_headboard": meta.get(
                    "headstone_has_photo_on_headboard"
                ),
                "headstone_photo_type": meta.get("headstone_photo_type"),
                "headstone_headboard_shape": meta.get("headstone_headboard_shape"),
                "headstone_cover_type": meta.get("headstone_cover_type"),
                "headstone_accessories": meta.get("headstone_accessories"),
                "category": meta.get("category", ""),
                "tags": meta.get("tags", ""),
            }
        )
    return out[:limit]


def _embed_query(text: str, model: str = "text-embedding-3-small") -> list[float]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty query")
    resp = client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding


def _get_coll(name: str):
    cli = PersistentClient(path=DB_DIR)
    return cli.get_collection(name)


def kb_search(query: str, k: int = 5):
    """
    Zwraca surowe trafienia z kolekcji knowledge_base:
    [{id, document, metadata, distance}, ...]
    """
    coll = _get_coll(KB_COLL)
    emb = _embed_query(query)
    res = coll.query(query_embeddings=[emb], n_results=max(1, int(k)))

    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = (res.get("distances") or [[None]])[0]
    for i in range(len(ids)):
        out.append(
            {
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i],
                "distance": dists[i] if i < len(dists) else None,
            }
        )
    return out
