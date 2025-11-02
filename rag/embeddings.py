# embeddings.py (utwórz i użyj w search/build_index)
import os
from functools import lru_cache
from typing import Iterable, List

PROVIDER = os.getenv("EMBED_PROVIDER", "sbert").strip().lower()

DEFAULT_MODELS = {
    "sbert": "intfloat/multilingual-e5-small",
    "gemini": "text-embedding-004",
    "openai": "text-embedding-3-small",
}

MODEL_ID = os.getenv("EMBED_MODEL_ID", DEFAULT_MODELS.get(PROVIDER, "unknown"))

_PREFIXES = {
    "sbert": {"query": "query: ", "document": "passage: "},
    "gemini": {"query": "query: ", "document": "document: "},
}


def _normalize_inputs(texts: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for item in texts:
        text = str(item or "").strip()
        if text:
            cleaned.append(text)
    if not cleaned:
        raise ValueError("embed_texts: empty input")
    return cleaned


def _apply_prefixes(texts: Iterable[str], purpose: str) -> List[str]:
    prefix = _PREFIXES.get(PROVIDER, {}).get(purpose)
    if not prefix:
        return list(texts)
    out: List[str] = []
    for t in texts:
        lowered = t.lower()
        if lowered.startswith(prefix.lower()):
            out.append(t)
        else:
            out.append(prefix + t)
    return out


@lru_cache(maxsize=1)
def _sbert():
    from sentence_transformers import SentenceTransformer

    # Najlepszy balans PL/jakość/CPU:
    return SentenceTransformer(MODEL_ID)


def embed_texts(texts, *, purpose: str = "document") -> List[List[float]]:
    """
    Zwraca listę wektorów dla podanych tekstów. Parametr `purpose` pozwala
    dostosować prefiksy wymagane przez poszczególne modele retrival (query/document).
    """
    if MODEL_ID == "unknown":
        raise ValueError(
            f"embed_texts: no model configured for provider '{PROVIDER}'. "
            "Set EMBED_MODEL_ID or choose a supported provider (sbert/gemini/openai)."
        )

    if isinstance(texts, str):
        texts = [texts]

    cleaned = _normalize_inputs(texts)
    prepared = _apply_prefixes(cleaned, purpose)

    if PROVIDER == "sbert":
        model = _sbert()
        return model.encode(prepared, normalize_embeddings=True).tolist()

    if PROVIDER == "gemini":
        import google.genai as genai

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        out: List[List[float]] = []
        for content in prepared:
            task_type = None
            if purpose == "query":
                task_type = "retrieval_query"
            elif purpose == "document":
                task_type = "retrieval_document"
            config = {"task_type": task_type} if task_type else None
            resp = client.models.embed_content(
                model=MODEL_ID,
                contents=content,
                config=config,
            )
            out.append(resp.embedding.values)
        return out

    if PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.embeddings.create(model=MODEL_ID, input=prepared)
        return [d.embedding for d in resp.data]

    raise ValueError(f"Unknown EMBED_PROVIDER={PROVIDER}")


def embed_query(text: str) -> List[float]:
    return embed_texts(text, purpose="query")[0]


def embed_documents(texts) -> List[List[float]]:
    return embed_texts(texts, purpose="document")


def embedding_fingerprint() -> str:
    return f"{PROVIDER}:{MODEL_ID}"
