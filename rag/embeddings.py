# embeddings.py (utwórz i użyj w search/build_index)
import os
from functools import lru_cache

PROVIDER = os.getenv("EMBED_PROVIDER", "sbert")  # sbert | gemini | openai


@lru_cache(maxsize=1)
def _sbert():
    from sentence_transformers import SentenceTransformer

    # Najlepszy balans PL/jakość/CPU:
    return SentenceTransformer("intfloat/multilingual-e5-small")


def embed_texts(texts):
    if isinstance(texts, str):
        texts = [texts]

    if PROVIDER == "sbert":
        model = _sbert()
        # e5 oczekuje prefiksów zadań; dla search dodajemy 'query: '
        return model.encode(texts, normalize_embeddings=True).tolist()

    elif PROVIDER == "gemini":
        import google.genai as genai
        from google.genai.types import EmbedContentRequest

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        out = []
        for t in texts:
            r = client.models.embed_content(
                model="text-embedding-004",
                request=EmbedContentRequest(content="query: " + t),
            )
            out.append(r.embedding.values)
        return out

    elif PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [d.embedding for d in resp.data]

    else:
        raise ValueError(f"Unknown EMBED_PROVIDER={PROVIDER}")
