### 1. Project Description

**Multi-Source Hybrid RAG Assistant**
A powerful RAG pipeline that harmonizes data from Microsoft SQL Server databases and local document stores (PDF/TXT). It leverages FAISS for high-speed vector search and SQLite for relational metadata management. By combining these sources, the assistant can answer complex queries about organizational data (like employee records) and textual knowledge bases (like manuals or books) using the Gemini 2.5 Flash model.

---

### 2. README.md

```markdown
# Hybrid Multi-Source RAG Assistant

A Retrieval-Augmented Generation (RAG) system capable of querying both **SQL Databases** and **Unstructured Documents** (PDF, TXT) in a single conversational thread.

## 🚀 Features
- **Hybrid Retrieval:** Merges search results from SQL Server tables and local PDF/TXT files.
- **Dual Vector Storage:** Uses **FAISS** for document embeddings and **SQLite** for structured data chunking and metadata.
- **LLM Integration:** Powered by Google Gemini (2.5-Flash) for high-quality, context-aware responses.
- **Interactive UIs:** Includes both a lightweight **Command Line Interface (CLI)** and a modern **Gradio Web UI**.
- **Conversation Memory:** Tracks short-term context to handle follow-up questions effectively.

## 🛠️ Architecture
1. **Extraction:** Pulls data from SQL Server (Employees/Offices) and local files.
2. **Embedding:** Generates 384-dimensional vectors using `all-MiniLM-L6-v2`.
3. **Indexing:** Stores document vectors in FAISS and SQL data in a local SQLite DB.
4. **Retrieval:** Performs a similarity search across both storage engines based on user intent.
5. **Generation:** Passes retrieved context and chat history to Gemini for the final answer.

## 📋 Prerequisites
- Python 3.9+
- Microsoft SQL Server (for the database extraction module)
- Google AI API Key

## ⚙️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/hybrid-rag-project.git](https://github.com/yourusername/hybrid-rag-project.git)
   cd hybrid-rag-project

```

2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Environment Variables:**
Create a `.env` file in the root directory and add your API key:
```env
api_key=YOUR_GOOGLE_GEMINI_API_KEY

```


4. **Initialize the SQL Pipeline:**
Ensure your SQL Server is running and updated in `sql_database.py`, then run:
```bash
python sql_database.py

```



## 🖥️ Usage

### Option 1: Web Interface (Recommended)

```bash
python gradio_ui.py

```

*Access the UI at `http://127.0.0.1:7860*`

### Option 2: CLI Chatbot

```bash
python RAG.py

```

## 📂 Project Structure

* `database.py`: Handles PDF/TXT extraction and FAISS indexing.
* `sql_database.py`: Manages SQL Server connection and SQLite embedding storage.
* `Query_data.py`: The logic engine that merges search results and calls Gemini.
* `gradio_ui.py`: The web-based frontend.

```
