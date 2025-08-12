# rag/search.py
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from chromadb import Client
from chromadb import PersistentClient

# from chromadb.config import Settings
from openai import OpenAI

# Ścieżki niezależne od bieżącego katalogu
ROOT = Path(__file__).resolve().parents[1]
DB_DIR = str(ROOT / "data" / "chroma_db")

GALLERY_COLL = "gallery"
ALLOWED_FILTERS = {"grave_type", "plaque_color", "stone", "finish", "category"}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed(q: str):
    return (
        client.embeddings.create(model="text-embedding-3-small", input=[q])
        .data[0]
        .embedding
    )


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
        if k in ALLOWED_FILTERS and v not in (None, "", []):
            clauses.append({k: {"$eq": v}})
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
                "size_w_cm": meta.get("size_w_cm"),
                "size_l_cm": meta.get("size_l_cm"),
                "category": meta.get("category", "nagrobki"),
                "tags": meta.get("tags", ""),
            }
        )
    return out[:limit]
