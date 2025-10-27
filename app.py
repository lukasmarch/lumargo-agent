import os, json, sqlite3, hashlib, glob
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

from rag.build_index import build_all
from rag.search import search_gallery
from rag.qa import answer_kb
from rag.search import kb_search

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


def get_conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    os.makedirs("data", exist_ok=True)
    with get_conn() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS sessions(
            session_id TEXT PRIMARY KEY,
            user_handle TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_handle TEXT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS leads(
            id INTEGER PRIMARY KEY,
            ts TEXT,
            name TEXT,
            phone TEXT,
            email TEXT,
            topic TEXT,
            preferences TEXT,
            notes TEXT,
            session_id TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE SET NULL
        );"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS appointments(
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            contact_id INTEGER,
            requested_slot TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE SET NULL,
            FOREIGN KEY(contact_id) REFERENCES leads(id) ON DELETE SET NULL
        );"""
        )

        # Migrations for existing deployments
        lead_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(leads)").fetchall()
        }
        if "session_id" not in lead_columns:
            conn.execute("ALTER TABLE leads ADD COLUMN session_id TEXT")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session_ts ON messages(session_id, ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_leads_session ON leads(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_appointments_session ON appointments(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_appointments_status_slot ON appointments(status, requested_slot)"
        )


init_db()


def ensure_session(conn: sqlite3.Connection, session_id: str, user_handle: Optional[str] = None):
    session_id = (session_id or "").strip()
    if not session_id:
        return
    conn.execute(
        "INSERT OR IGNORE INTO sessions(session_id, user_handle) VALUES(?, ?)",
        (session_id, user_handle),
    )
    if user_handle:
        conn.execute(
            "UPDATE sessions SET user_handle=? WHERE session_id=? AND (user_handle IS NULL OR user_handle='')",
            (user_handle, session_id),
        )


def fetch_session(conn: sqlite3.Connection, session_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT session_id, user_handle, created_at FROM sessions WHERE session_id=?",
        (session_id,),
    ).fetchone()


def fetch_messages(conn: sqlite3.Connection, session_id: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """SELECT session_id, user_handle, role, content, ts
        FROM messages
        WHERE session_id=?
        ORDER BY ts ASC, id ASC""",
        (session_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def save_lead(
    name: str,
    phone: str | None,
    email: str | None,
    topic: str,
    preferences: dict,
    notes: str | None,
    session_id: Optional[str] = None,
    user_handle: Optional[str] = None,
):
    with get_conn() as conn:
        if session_id:
            ensure_session(conn, session_id, user_handle)
        cursor = conn.execute(
            """INSERT INTO leads(
            ts,name,phone,email,topic,preferences,notes,session_id
        ) VALUES(datetime('now'),?,?,?,?,?,?,?)""",
            (
                name,
                phone,
                email,
                topic,
                json.dumps(preferences, ensure_ascii=False),
                notes,
                session_id,
            ),
        )
        lead_id = cursor.lastrowid
        if session_id and lead_id:
            conn.execute(
                """UPDATE appointments
                   SET contact_id = COALESCE(contact_id, ?)
                 WHERE session_id = ? AND contact_id IS NULL""",
                (lead_id, session_id),
            )
        return lead_id


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
    session_id: Optional[str] = None
    user_handle: Optional[str] = None


class MemoryAppendReq(BaseModel):
    role: str
    content: str
    user_handle: Optional[str] = None
    ts: Optional[str] = None


class AppointmentCreateReq(BaseModel):
    session_id: Optional[str] = None
    contact_id: Optional[int] = None
    requested_slot: str
    status: Optional[str] = "pending"
    notes: Optional[str] = None
    user_handle: Optional[str] = None


class AppointmentUpdateReq(BaseModel):
    requested_slot: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None
    contact_id: Optional[int] = None


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


@app.get("/debug/peek")
def debug_peek(q: str, k: int = 5):
    hits = kb_search(q, k=k)  # zwróć np. metadatas, documents, distances
    return {"query": q, "hits": hits}


@app.post("/search")
def api_search(req: SearchReq):
    results = search_gallery(req.query, req.filters or {}, req.limit)
    return {"results": results}


@app.post("/answer")
def api_answer(req: AnswerReq):
    return answer_kb(req.question)


@app.post("/lead")
def api_lead(req: LeadReq):
    lead_id = save_lead(
        req.name,
        req.phone,
        req.email,
        req.topic,
        req.preferences or {},
        req.notes,
        req.session_id,
        req.user_handle,
    )
    return {"ok": True, "lead_id": lead_id}


@app.get("/memory/{session_id}")
def api_memory(session_id: str):
    session_id = (session_id or "").strip()
    if not session_id:
        raise HTTPException(400, "session_id is required")
    with get_conn() as conn:
        session_row = fetch_session(conn, session_id)
        if session_row is None:
            raise HTTPException(404, "session not found")
        messages = fetch_messages(conn, session_id)
    return {
        "session_id": session_row["session_id"],
        "user_handle": session_row["user_handle"],
        "created_at": session_row["created_at"],
        "messages": messages,
    }


@app.post("/memory/{session_id}")
def api_memory_append(session_id: str, entry: MemoryAppendReq):
    session_id = (session_id or "").strip()
    if not session_id:
        raise HTTPException(400, "session_id is required")
    role = (entry.role or "").strip()
    if not role:
        raise HTTPException(400, "role is required")
    content = (entry.content or "").strip()
    if not content:
        raise HTTPException(400, "content is required")
    with get_conn() as conn:
        ensure_session(conn, session_id, entry.user_handle)
        cursor = conn.execute(
            """INSERT INTO messages(session_id, user_handle, role, content, ts)
               VALUES(?,?,?,?, COALESCE(?, CURRENT_TIMESTAMP))""",
            (session_id, entry.user_handle, role, content, entry.ts),
        )
        message_id = cursor.lastrowid
        row = conn.execute(
            """SELECT session_id, user_handle, role, content, ts
               FROM messages WHERE id=?""",
            (message_id,),
        ).fetchone()
    return dict(row)


@app.get("/appointments/availability")
def api_appointments_availability(days: int = 7, slots_per_day: int = 2):
    days = max(1, min(days, 30))
    slots_per_day = max(1, min(slots_per_day, 4))
    base_times = ["09:00", "12:00", "15:00", "17:00"]
    now = datetime.utcnow()
    with get_conn() as conn:
        taken_rows = conn.execute(
            """SELECT requested_slot FROM appointments
               WHERE status IN ('pending','confirmed')"""
        ).fetchall()
    taken = {row["requested_slot"] for row in taken_rows if row["requested_slot"]}
    slots: List[str] = []
    for offset in range(days):
        day = (now + timedelta(days=offset)).date()
        used = 0
        for time_str in base_times:
            if used >= slots_per_day:
                break
            slot = f"{day}T{time_str}"
            if slot in taken:
                continue
            slots.append(slot)
            used += 1
    return {"slots": slots}


@app.post("/appointments")
def api_appointments_create(req: AppointmentCreateReq):
    slot = (req.requested_slot or "").strip()
    if not slot:
        raise HTTPException(400, "requested_slot is required")
    status = (req.status or "pending").strip() or "pending"
    session_id = (req.session_id or "").strip() or None
    with get_conn() as conn:
        if session_id:
            ensure_session(conn, session_id, req.user_handle)
        cursor = conn.execute(
            """INSERT INTO appointments(session_id, contact_id, requested_slot, status, notes)
               VALUES(?,?,?,?,?)""",
            (session_id, req.contact_id, slot, status, req.notes),
        )
        appointment_id = cursor.lastrowid
        conn.execute(
            "UPDATE appointments SET updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (appointment_id,),
        )
        row = conn.execute(
            """SELECT id, session_id, contact_id, requested_slot, status, notes,
                      created_at, updated_at
               FROM appointments WHERE id=?""",
            (appointment_id,),
        ).fetchone()
    return dict(row)


@app.patch("/appointments/{appointment_id}")
def api_appointments_update(appointment_id: int, req: AppointmentUpdateReq):
    updates = []
    params: List[Any] = []
    if req.requested_slot is not None:
        slot = req.requested_slot.strip()
        if not slot:
            raise HTTPException(400, "requested_slot cannot be empty")
        updates.append("requested_slot=?")
        params.append(slot)
    if req.status is not None:
        status = req.status.strip()
        if not status:
            raise HTTPException(400, "status cannot be empty")
        updates.append("status=?")
        params.append(status)
    if req.notes is not None:
        updates.append("notes=?")
        params.append(req.notes)
    if req.contact_id is not None:
        updates.append("contact_id=?")
        params.append(req.contact_id)
    if not updates:
        raise HTTPException(400, "no updates supplied")
    params.append(appointment_id)
    with get_conn() as conn:
        cursor = conn.execute(
            f"""UPDATE appointments
                SET {', '.join(updates)}, updated_at=CURRENT_TIMESTAMP
              WHERE id=?""",
            params,
        )
        if cursor.rowcount == 0:
            raise HTTPException(404, "appointment not found")
        row = conn.execute(
            """SELECT id, session_id, contact_id, requested_slot, status, notes,
                      created_at, updated_at
               FROM appointments WHERE id=?""",
            (appointment_id,),
        ).fetchone()
    return dict(row)


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
