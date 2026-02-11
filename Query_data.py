import os
import json
import sqlite3
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("api_key"))
MODEL_NAME = "gemini-2.5-flash"

def search_sql_database(query_text, model, json_path="database/database.json", db_path="database/database.db", k=3):
    unique_entries = {}

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        for entry in json_data:
            key = (entry.get("source"), entry.get("chunk"))
            unique_entries[key] = np.array(entry["vector"], dtype=np.float32)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
            
    query = """
        SELECT d.source, c.chunk_text, e.vector
        FROM documents d
        JOIN chunks c ON d.id = c.document_id
        JOIN embeddings e ON c.id = e.chunk_id
    """
    cur.execute(query)
    rows = cur.fetchall()
    conn.close()

    for src, text, vec_blob in rows:
        vector = np.frombuffer(vec_blob, dtype=np.float32)
        key = (src, text)
        unique_entries[key] = vector

    query_vector = model.encode([query_text], normalize_embeddings=True)[0]

    results = []
    for (source, text), vector in unique_entries.items():
        score = np.dot(query_vector, vector)
        
        results.append({
            "source": source,
            "text": text,
            "score": score
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]


def query_rag(query_text, db, conversation_memory=""):
    context_parts = []
    sources = set()

    file_docs = db.retrieve(query_text, k=3)

    sql_docs = search_sql_database(
        query_text, 
        db.model, 
        json_path="database/database.json", 
        db_path="database/database.db", 
        k=3
    )

    all_docs = file_docs + sql_docs

    for doc in all_docs:
        source_name = doc.get("source", "Unknown")
        sources.add(source_name)
        text_content = doc.get("text") or doc.get("chunk")
        context_parts.append(f"[SOURCE: {source_name}]\n{text_content}")

    if context_parts:
        context_text = "\n\n---\n\n".join(context_parts)
    else:
        context_text = "No relevant documents found in the database."

    final_prompt = (
        f"Context:\n{context_text}\n\n"
        f"History:\n{conversation_memory}\n\n"
        f"Question: {query_text}"
    )
 
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(final_prompt)
    return response.text, list(sources)
