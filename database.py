import os
import faiss
import json
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

class VectorDatabase:
    def __init__(self, model_name="all-MiniLM-L6-v2", index_file="vector_store.index", docs_file="documents.json"):
        self.model = SentenceTransformer(model_name, local_files_only=True)
        self.index_file = index_file
        self.docs_file = docs_file
        self.embed_dim = 384
        
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.docs_file, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.embed_dim)
            self.documents = []

    def extract_text(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            reader = PdfReader(file_path)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        return ""

    def chunk_text(self, text, chunk_size=500, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def ingest_files(self, file_paths):
        for path in file_paths:
            if not os.path.exists(path): continue
            text = self.extract_text(path)
            chunks = self.chunk_text(text)
            embeddings = self.model.encode(chunks, normalize_embeddings=True)
            start_id = len(self.documents)
            self.index.add(np.array(embeddings).astype('float32'))
            for i, chunk in enumerate(chunks):
                self.documents.append({"id": start_id + i, "text": chunk, "source": os.path.basename(path)})
        self.save()

    def save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.docs_file, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2)

    def retrieve(self, query, k=3):
        query_vector = self.model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        return [self.documents[idx] for idx in indices[0] if idx < len(self.documents)]


db = VectorDatabase()

files_to_process = [
    "data/Sherlock Holmes.txt",
    "data/Alice_in_Wonderland.pdf",
    "data/402164=7039-Bryson, Bill - A Short History of Nearly Everything.pdf"
]
    
if not db.documents:
    db.ingest_files(files_to_process)

# query = "Who is Sherlock Holmes?"
# results = db.retrieve(query)

# print(f"\nResults for: '{query}'")
# for r in results:
#     print("-" * 30)
#     print(f"SOURCE: {r['source']}")
#     print(f"TEXT: {r['text'][:200]}...")