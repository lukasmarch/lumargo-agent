# === BAZA ===
FROM python:3.11-slim-bookworm

# --- Ustawienia środowiska ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1

WORKDIR /app

# --- Instalacja niezbędnych pakietów systemowych ---
# build-essential potrzebny m.in. dla chromadb / duckdb
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Instalacja ultra-szybkiego menedżera pakietów uv ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# --- Kopiowanie pliku zależności ---
COPY requirements.txt /app/requirements.txt

# --- Instalacja zależności przy użyciu uv ---
RUN uv pip install --system -r /app/requirements.txt

# --- Kopiowanie całego kodu aplikacji ---
COPY . /app

# --- Utworzenie katalogu na dane (Render Disk) ---
RUN mkdir -p /app/data/chroma_db

# --- Ekspozycja portu (Render używa zmiennej PORT) ---
EXPOSE 8000

# --- Komenda startowa ---
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]

