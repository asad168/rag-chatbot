import os
import numpy as np
import json
import sqlite3
import pyodbc
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  

def get_sql_connection():
    return pyodbc.connect(
        'DRIVER={SQL Server};'
        'SERVER=DESKTOP-TCRP0DB;'
        'DATABASE=KxP_hr;'
        'Trusted_Connection=yes;'
    )


def get_sqlite_connection():
    conn = sqlite3.connect("database/database.db")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_sqlite_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT UNIQUE
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        chunk_text TEXT,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_id INTEGER,
        vector BLOB,
        FOREIGN KEY(chunk_id) REFERENCES chunks(id)
    )
    """)

    conn.commit()
    conn.close()


def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []

    step = max(1, chunk_size - overlap)

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def embed_texts(texts):
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True 
    )
    return vectors.astype(np.float32)


def insert_document(source):
    conn = get_sqlite_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT OR IGNORE INTO documents (source) VALUES (?)",
        (source,)
    )

    cur.execute("SELECT id FROM documents WHERE source=?", (source,))
    doc_id = cur.fetchone()[0]

    conn.commit()
    conn.close()
    return doc_id


def save_chunks(document_id, chunks, vectors):
    conn = get_sqlite_connection()
    cur = conn.cursor()

    for chunk, vector in zip(chunks, vectors):
        cur.execute(
            "INSERT INTO chunks (document_id, chunk_text) VALUES (?, ?)",
            (document_id, chunk)
        )
        cid = cur.lastrowid

        cur.execute(
            "INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)",
            (cid, vector.astype(np.float32).tobytes())
        )

    conn.commit()
    conn.close()


def extract_from_sql():
    conn = get_sql_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM employees")
    employees = cur.fetchall()

    cur.execute("SELECT * FROM offices")
    offices = cur.fetchall()

    conn.close()

    texts = []

    for e in employees:
        texts.append(f"Employee data: {e}")

    for o in offices:
        texts.append(f"Office data: {o}")

    return texts


def export_to_json(path="database/database.json"):
    os.makedirs("database", exist_ok=True)

    conn = get_sqlite_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT d.source, c.chunk_text, e.vector
        FROM documents d
        JOIN chunks c ON d.id = c.document_id
        JOIN embeddings e ON c.id = e.chunk_id
    """)

    rows = cur.fetchall()
    conn.close()

    data = [
        {
            "source": src,
            "chunk": chunk,
            "vector": np.frombuffer(vec, dtype=np.float32).tolist()
        }
        for src, chunk, vec in rows
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_pipeline():
    init_db()

    texts = extract_from_sql()

    doc_id = insert_document("sql_server_data")

    all_chunks = []
    for t in texts:
        all_chunks.extend(chunk_text(t))

    vectors = embed_texts(all_chunks)

    save_chunks(doc_id, all_chunks, vectors)

    export_to_json()

    print("RAG database ready.")


run_pipeline()
