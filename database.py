import sqlite3
import json
import os
import numpy as np

DB_PATH = os.path.join(os.path.dirname(__file__), "pdftota.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Create documents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE,
        page_count INTEGER,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Create document_chunks table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        page_number INTEGER,
        chunk_index INTEGER,
        text TEXT,
        embedding BLOB,
        FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
    );
    """)
    
    # Create chat_sessions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id TEXT PRIMARY KEY,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Create chat_messages table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        sender TEXT,
        text TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        citations TEXT,
        FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
    );
    """)
    
    # Create settings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """)
    
    # Seed default settings
    env_gemini_key = ""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r") as f:
                for line in f:
                    if line.strip().startswith("GEMINI_API_KEY="):
                        env_gemini_key = line.split("=", 1)[1].strip()
                        break
        except Exception:
            pass

    default_settings = {
        "llm_provider": "ollama",
        "ollama_model": "pdftota",
        "gemini_api_key": env_gemini_key,
        "gemini_model": "gemini-2.5-flash",
        "openai_api_key": "",
        "openai_model": "gpt-4o-mini",
        "system_prompt": "You are a helpful and conversational PDF voice assistant named PDFtota. You can chat friendly with the user, answer questions using the provided PDF content when relevant, and cite pages (e.g., [Page X]) when referencing documents. If a query is generic or unrelated to the PDFs, chat normally without document constraints.",
        "temperature": "0.3",
        "chunk_size": "800",
        "chunk_overlap": "150"
    }
    
    for key, value in default_settings.items():
        cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?);", (key, value))
        
    # Migration: Update old system prompt if not customized by user
    cursor.execute(
        "UPDATE settings SET value = ? WHERE key = 'system_prompt' AND value LIKE '%Answer questions accurately based on%';", 
        (default_settings["system_prompt"],)
    )

    # Migration: Update legacy gemini-1.5-pro model setting to gemini-2.5-pro
    cursor.execute("UPDATE settings SET value = 'gemini-2.5-pro' WHERE key = 'gemini_model' AND value = 'gemini-1.5-pro';")

    # Migration: Update existing empty gemini_api_key in db with the .env key if present
    if env_gemini_key:
        cursor.execute("UPDATE settings SET value = ? WHERE key = 'gemini_api_key' AND value = '';", (env_gemini_key,))
        
    conn.commit()
    conn.close()

# Document operations
def add_document(filename, page_count, content):
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO documents (filename, page_count, content) VALUES (?, ?, ?);",
            (filename, page_count, content)
        )
        doc_id = cursor.lastrowid
        conn.commit()
        return doc_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def update_document_stats(doc_id, page_count, content):
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE documents SET page_count = ?, content = ? WHERE id = ?;",
            (page_count, content, doc_id)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def delete_document(doc_id):
    conn = get_db()
    cursor = conn.cursor()
    try:
        # File cascades to document_chunks due to CASCADE delete setting (need PRAGMA enabled)
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("DELETE FROM documents WHERE id = ?;", (doc_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_documents():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, page_count, created_at FROM documents ORDER BY created_at DESC;")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_document_by_name(filename):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, page_count, content, created_at FROM documents WHERE filename = ?;", (filename,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# Chunk operations
def add_chunks(chunks_data):
    """
    chunks_data: list of dicts with:
      document_id, page_number, chunk_index, text, embedding (list of floats or None)
    """
    conn = get_db()
    cursor = conn.cursor()
    try:
        for chunk in chunks_data:
            embedding_blob = None
            if chunk.get("embedding") is not None:
                # Convert list to float32 numpy array and serialize as raw bytes
                arr = np.array(chunk["embedding"], dtype=np.float32)
                embedding_blob = arr.tobytes()
            
            cursor.execute(
                "INSERT INTO document_chunks (document_id, page_number, chunk_index, text, embedding) VALUES (?, ?, ?, ?, ?);",
                (chunk["document_id"], chunk["page_number"], chunk["chunk_index"], chunk["text"], embedding_blob)
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_all_chunks(active_doc_ids=None):
    conn = get_db()
    cursor = conn.cursor()
    
    if active_doc_ids:
        placeholders = ",".join("?" for _ in active_doc_ids)
        query = f"""
            SELECT c.id, c.document_id, c.page_number, c.chunk_index, c.text, c.embedding, d.filename
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.document_id IN ({placeholders})
        """
        cursor.execute(query, tuple(active_doc_ids))
    else:
        query = """
            SELECT c.id, c.document_id, c.page_number, c.chunk_index, c.text, c.embedding, d.filename
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
        """
        cursor.execute(query)
        
    rows = cursor.fetchall()
    conn.close()
    
    chunks = []
    for row in rows:
        d = dict(row)
        if d["embedding"] is not None:
            # Reconstruct float32 array
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
        chunks.append(d)
    return chunks

# Chat Operations
def create_session(session_id, title):
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO chat_sessions (id, title) VALUES (?, ?);",
            (session_id, title)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def rename_session(session_id, new_title):
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE chat_sessions SET title = ? WHERE id = ?;", (new_title, session_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_sessions():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, created_at FROM chat_sessions ORDER BY created_at DESC;")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_session(session_id):
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("DELETE FROM chat_sessions WHERE id = ?;", (session_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def add_message(session_id, sender, text, citations=None):
    conn = get_db()
    cursor = conn.cursor()
    try:
        citations_str = json.dumps(citations) if citations else None
        cursor.execute(
            "INSERT INTO chat_messages (session_id, sender, text, citations) VALUES (?, ?, ?, ?);",
            (session_id, sender, text, citations_str)
        )
        msg_id = cursor.lastrowid
        conn.commit()
        return msg_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_messages(session_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, sender, text, timestamp, citations FROM chat_messages WHERE session_id = ? ORDER BY id ASC;",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    messages = []
    for row in rows:
        d = dict(row)
        if d["citations"]:
            d["citations"] = json.loads(d["citations"])
        messages.append(d)
    return messages

# Settings Operations
def get_settings():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM settings;")
    rows = cursor.fetchall()
    conn.close()
    return {row["key"]: row["value"] for row in rows}

def update_settings(settings_dict):
    conn = get_db()
    cursor = conn.cursor()
    try:
        for key, value in settings_dict.items():
            cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?);", (key, str(value)))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
