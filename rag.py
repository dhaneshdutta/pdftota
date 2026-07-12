import os
import math
import re
import json
import numpy as np
import requests
import pdfplumber
from pypdf import PdfReader

# Text chunking with sliding window
def chunk_text(text, filename, doc_id, page_num, chunk_size=800, overlap=150):
    """
    Splits text into chunks of specified size and overlap.
    Returns a list of dicts containing chunk metadata.
    """
    chunks = []
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return chunks
        
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a space or punctuation to avoid cutting words/sentences
        if end < len(text):
            # Look backwards up to 100 characters for a natural boundary
            boundary = -1
            for i in range(end, max(start, end - 100), -1):
                if text[i] in ['.', '!', '?', '\n']:
                    boundary = i + 1
                    break
                elif text[i] == ' ' and boundary == -1:
                    boundary = i
            if boundary != -1:
                end = boundary
                
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "document_id": doc_id,
                "filename": filename,
                "page_number": page_num,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "embedding": None
            })
            chunk_index += 1
            
        start += (chunk_size - overlap)
        if start >= len(text) or (chunk_size - overlap) <= 0:
            break
            
    return chunks

def extract_pdf_chunks(filepath, doc_id, chunk_size=800, overlap=150):
    """
    Extracts text from a PDF page-by-page, chunks it, and returns the list of chunks.
    Also returns the total page count.
    """
    all_chunks = []
    filename = os.path.basename(filepath)
    page_count = 0
    
    # Try using pdfplumber first
    try:
        with pdfplumber.open(filepath) as pdf:
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    page_chunks = chunk_text(text, filename, doc_id, i + 1, chunk_size, overlap)
                    all_chunks.extend(page_chunks)
    except Exception as e:
        print(f"pdfplumber failed: {e}. Falling back to pypdf.")
        # Fallback to pypdf
        try:
            reader = PdfReader(filepath)
            page_count = len(reader.pages)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    page_chunks = chunk_text(text, filename, doc_id, i + 1, chunk_size, overlap)
                    all_chunks.extend(page_chunks)
        except Exception as e2:
            print(f"pypdf also failed: {e2}")
            raise Exception("Failed to parse PDF text.")
            
    return all_chunks, page_count

# Embedding Generation
def get_embedding(text, settings):
    """
    Generates embedding for a text chunk using the selected provider.
    Returns a list of floats, or None if embed generation fails.
    """
    provider = settings.get("llm_provider", "ollama")
    
    if provider == "gemini":
        api_key = settings.get("gemini_api_key")
        if not api_key:
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "models/text-embedding-004",
            "content": {
                "parts": [{"text": text}]
            }
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                res_data = response.json()
                return res_data.get("embedding", {}).get("values")
            else:
                print(f"Gemini embedding error: {response.text}")
        except Exception as e:
            print(f"Gemini embedding request exception: {e}")
            
    elif provider == "openai":
        api_key = settings.get("openai_api_key")
        if not api_key:
            return None
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": settings.get("openai_model", "text-embedding-3-small"),
            "input": text
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                res_data = response.json()
                return res_data.get("data", [{}])[0].get("embedding")
            else:
                print(f"OpenAI embedding error: {response.text}")
        except Exception as e:
            print(f"OpenAI embedding request exception: {e}")
            
    elif provider == "ollama":
        # Check if Ollama service is active
        url = "http://localhost:11434/api/embed"
        model = settings.get("ollama_model", "pdftota")
        payload = {
            "model": model,
            "input": text
        }
        try:
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                res_data = response.json()
                # Ollama returns list of embeddings for list input or a single list for string input
                embeddings = res_data.get("embeddings")
                if embeddings and len(embeddings) > 0:
                    return embeddings[0]
                elif "embedding" in res_data:
                    return res_data["embedding"]
            else:
                # Fallback to /api/embeddings
                fallback_url = "http://localhost:11434/api/embeddings"
                fallback_payload = {
                    "model": model,
                    "prompt": text
                }
                res2 = requests.post(fallback_url, json=fallback_payload, timeout=15)
                if res2.status_code == 200:
                    return res2.json().get("embedding")
        except Exception as e:
            print(f"Ollama embedding exception: {e}")
            
    return None

# Fallback TF-IDF for offline/error states
def compute_tfidf_similarity(query, chunks, top_k=5):
    """
    Pure Python/NumPy TF-IDF similarity fallback.
    Returns chunks sorted by similarity score.
    """
    def tokenize(s):
        return [w.lower() for w in re.findall(r'\w+', s) if len(w) > 2]
        
    stop_words = {"the", "and", "of", "to", "a", "in", "is", "that", "it", "on", "for", "with", "as", "was", "at", "by", "an"}
    
    # Tokenize documents
    doc_tokens = [tokenize(c["text"]) for c in chunks]
    query_tokens = tokenize(query)
    
    # Vocabulary
    vocab = set()
    for tokens in doc_tokens:
        vocab.update(tokens)
    vocab.update(query_tokens)
    vocab = list(vocab)
    vocab_idx = {word: i for i, word in enumerate(vocab)}
    
    if not vocab or not chunks:
        return []
        
    # Document Frequency (DF)
    df = {}
    for tokens in doc_tokens:
        seen = set(tokens)
        for token in seen:
            df[token] = df.get(token, 0) + 1
            
    # IDF
    n_docs = len(chunks)
    idf = {}
    for word in vocab:
        df_w = df.get(word, 0)
        # Add 1 smoothing
        idf[word] = math.log((n_docs + 1) / (df_w + 1)) + 1
        
    # TF-IDF vectors
    def get_tfidf_vec(tokens):
        vec = np.zeros(len(vocab), dtype=np.float32)
        tf = {}
        for token in tokens:
            if token not in stop_words:
                tf[token] = tf.get(token, 0) + 1
        for token, count in tf.items():
            if token in vocab_idx:
                idx = vocab_idx[token]
                vec[idx] = count * idf[token]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
        
    doc_vectors = [get_tfidf_vec(tokens) for tokens in doc_tokens]
    query_vector = get_tfidf_vec(query_tokens)
    
    # Compute similarity (dot product since normalized)
    scores = []
    for idx, doc_vec in enumerate(doc_vectors):
        score = float(np.dot(query_vector, doc_vec))
        scores.append((score, chunks[idx]))
        
    # Sort and take top_k
    scores.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, chunk in scores[:top_k]:
        chunk_copy = chunk.copy()
        chunk_copy["score"] = score
        results.append(chunk_copy)
    return results

# Retrieval pipeline
def retrieve_contexts(query, all_chunks, settings, top_k=5):
    """
    Retrieves the most similar chunks for a query.
    Falls back to TF-IDF if embeddings are unavailable or missing.
    """
    if not all_chunks:
        return []
        
    # Check if we have embeddings for the chunks
    has_embeddings = all(c.get("embedding") is not None for c in all_chunks)
    
    if has_embeddings:
        query_embedding = get_embedding(query, settings)
        if query_embedding is not None:
            query_vec = np.array(query_embedding, dtype=np.float32)
            # Normalize query vector
            q_norm = np.linalg.norm(query_vec)
            if q_norm > 0:
                query_vec = query_vec / q_norm
                
            scores = []
            for chunk in all_chunks:
                chunk_vec = chunk["embedding"]
                c_norm = np.linalg.norm(chunk_vec)
                if c_norm > 0:
                    normalized_chunk_vec = chunk_vec / c_norm
                    score = float(np.dot(query_vec, normalized_chunk_vec))
                else:
                    score = 0.0
                scores.append((score, chunk))
                
            scores.sort(key=lambda x: x[0], reverse=True)
            results = []
            for score, chunk in scores[:top_k]:
                chunk_copy = chunk.copy()
                # Remove large numpy array from dict for response cleanliness
                chunk_copy.pop("embedding", None)
                chunk_copy["score"] = score
                results.append(chunk_copy)
            return results
            
    # TF-IDF Fallback
    print("Falling back to pure Python TF-IDF search")
    return compute_tfidf_similarity(query, all_chunks, top_k)

# Chat inference wrappers
def chat_completion(prompt, settings):
    """
    Wrapper for chat generation using selected provider.
    Returns response text.
    """
    provider = settings.get("llm_provider", "ollama")
    temperature = float(settings.get("temperature", 0.3))
    system_prompt = settings.get("system_prompt", "")
    
    if provider == "gemini":
        api_key = settings.get("gemini_api_key")
        model = settings.get("gemini_model", "gemini-2.5-flash")
        if not api_key:
            return "Error: Gemini API key is missing. Please add it in Settings."
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "generationConfig": {
                "temperature": temperature
            }
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                res_data = response.json()
                try:
                    return res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                except (KeyError, IndexError):
                    return "Error parsing Gemini response."
            else:
                return f"Gemini API returned status code {response.status_code}: {response.text}"
        except Exception as e:
            return f"Error communicating with Gemini: {str(e)}"
            
    elif provider == "openai":
        api_key = settings.get("openai_api_key")
        model = settings.get("openai_model", "gpt-4o-mini")
        if not api_key:
            return "Error: OpenAI API key is missing. Please add it in Settings."
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                res_data = response.json()
                return res_data["choices"][0]["message"]["content"].strip()
            else:
                return f"OpenAI API returned status code {response.status_code}: {response.text}"
        except Exception as e:
            return f"Error communicating with OpenAI: {str(e)}"
            
    elif provider == "ollama":
        url = "http://localhost:11434/api/generate"
        model = settings.get("ollama_model", "pdftota")
        
        full_prompt = f"System: {system_prompt}\n\nContext and Question:\n{prompt}"
        payload = {
            "model": model,
            "prompt": full_prompt,
            "options": {
                "temperature": temperature
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            response_text = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    response_json = json.loads(decoded_line)
                    if 'response' in response_json:
                        response_text += response_json['response']
            return response_text.strip()
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {str(e)}. Make sure Ollama is running (`ollama serve`)."
            
    return "Error: Unsupported LLM provider configuration."
