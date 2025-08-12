import os, json, sqlite3, hashlib, glob
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

from rag.build_index import build_all
from rag.search import search_gallery
from rag.qa import answer_kb

from chromadb import Client

# from chromadb.config import Settings
from chromadb import PersistentClient
from pathlib import Path


from dotenv import load_dotenv

load_dotenv()  # wczyta plik .env z katalogu projektu, jeśli istnieje

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI(title="Lumargo Agent (RAG)")

# --- CORS: dopuszczamy Twoją domenę (dodaj dev origin przy testach)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lumargo.pl",
        "https://www.lumargo.pl",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Statyczne pliki (obrazy)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Auto-reindex (gdy zmieni się gallery.csv lub pliki w KB)
STAMP = "data/.rag.md5"


def current_stamp():
    h = hashlib.md5()
    with open("data/gallery.csv", "rb") as f:
        h.update(f.read())
    for fp in sorted(glob.glob("data/kb/*.*")):
        with open(fp, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


@app.on_event("startup")
def maybe_reindex():
    os.makedirs("data/chroma_db", exist_ok=True)
    new = current_stamp()
    old = ""
    if os.path.exists(STAMP):
        with open(STAMP) as f:
            old = f.read().strip()
    if new != old:
        print("[RAG] content changed → rebuilding index")
        build_all()
        with open(STAMP, "w") as f:
            f.write(new)
    else:
        print("[RAG] index up-to-date")


# --- LEADS (SQLite prosto)
DB = "data/leads.db"


def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS leads(
        id INTEGER PRIMARY KEY,
        ts TEXT,
        name TEXT, phone TEXT, email TEXT,
        topic TEXT,                   -- 'nagrobek' / 'blat' / inne
        preferences TEXT,             -- JSON (grave_type/stone/finish/...)
        notes TEXT
    );"""
    )
    conn.commit()
    conn.close()


init_db()


def save_lead(
    name: str,
    phone: str | None,
    email: str | None,
    topic: str,
    preferences: dict,
    notes: str | None,
):
    conn = sqlite3.connect(DB)
    conn.execute(
        "INSERT INTO leads(ts,name,phone,email,topic,preferences,notes) VALUES(datetime('now'),?,?,?,?,?,?)",
        (name, phone, email, topic, json.dumps(preferences, ensure_ascii=False), notes),
    )
    conn.commit()
    conn.close()


# --- Schemy requestów
class SearchReq(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 3


class AnswerReq(BaseModel):
    question: str


class LeadReq(BaseModel):
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None
    topic: str  # np. "nagrobek" / "blat"
    preferences: Dict[
        str, Any
    ]  # np. {"grave_type":"pojedynczy","plaque_color":"czarna"}
    notes: Optional[str] = None


# --- Endpointy


@app.get("/debug/status")
def debug_status():
    DB = str((Path(__file__).resolve().parent / "data" / "chroma_db").resolve())
    try:
        coll = PersistentClient(path=DB).get_collection("gallery")
        data = coll.get(include=["metadatas"])  # "ids" wracają zawsze
        n = len(data.get("ids", []))
        sample = data.get("metadatas", [None])[0]
    except Exception as e:
        n, sample = 0, str(e)
    return {"db_dir": DB, "gallery_items": n, "sample": sample}


@app.post("/search")
def api_search(req: SearchReq):
    results = search_gallery(req.query, req.filters or {}, req.limit)
    return {"results": results}


@app.post("/answer")
def api_answer(req: AnswerReq):
    return answer_kb(req.question)


@app.post("/lead")
def api_lead(req: LeadReq):
    save_lead(req.name, req.phone, req.email, req.topic, req.preferences, req.notes)
    return {"ok": True}


ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")


@app.post("/admin/reindex")
def api_reindex(token: str):
    if token != ADMIN_TOKEN:
        raise HTTPException(401, "unauthorized")
    build_all()
    return {"ok": True}


@app.get("/")
def root():
    return {"ok": True}
