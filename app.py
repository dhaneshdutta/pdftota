import os
import uuid
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory
import speech_recognition as sr
import tempfile
from gtts import gTTS

import database
import rag

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Database Schema
database.init_db()

@app.route('/')
def index():
    return render_template('index.html')

# Settings Endpoint
@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    if request.method == 'GET':
        return jsonify(database.get_settings()), 200
    else:
        new_settings = request.json
        if not new_settings:
            return jsonify({"error": "Invalid payload"}), 400
        database.update_settings(new_settings)
        return jsonify({"message": "Settings updated successfully"}), 200

# Ollama Models Auto-detection
@app.route('/api/ollama/models', methods=['GET'])
def api_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models_data = response.json()
            models = [m["name"] for m in models_data.get("models", [])]
            return jsonify({"models": models}), 200
    except Exception:
        pass
    return jsonify({"models": []}), 200

# Document Upload & List
@app.route('/api/documents', methods=['GET', 'POST'])
def api_documents():
    if request.method == 'GET':
        return jsonify(database.get_documents()), 200
        
    elif request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        if file and file.filename.lower().endswith('.pdf'):
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            # Secure unique name if needed, but keeping original name makes it easier for citations
            # Let's check if document already exists
            existing_doc = database.get_document_by_name(filename)
            if existing_doc:
                # Remove it first to re-index it
                database.delete_document(existing_doc["id"])
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
                        
            file.save(filepath)
            
            try:
                # Insert initial doc entry to get database ID
                doc_id = database.add_document(filename, 0, "")
                
                # Fetch settings for chunk configurations
                settings = database.get_settings()
                chunk_size = int(settings.get("chunk_size", 800))
                chunk_overlap = int(settings.get("chunk_overlap", 150))
                
                # Extract and chunk PDF content
                chunks, page_count = rag.extract_pdf_chunks(filepath, doc_id, chunk_size, chunk_overlap)
                
                # Update page count and content summary in DB
                full_content = " ".join([c["text"] for c in chunks])
                database.update_document_stats(doc_id, page_count, full_content)
                
                # Calculate embeddings for each chunk
                print(f"Generating embeddings for {len(chunks)} chunks in {filename}...")
                for chunk in chunks:
                    chunk["embedding"] = rag.get_embedding(chunk["text"], settings)
                
                # Save chunks to database
                database.add_chunks(chunks)
                
                return jsonify({
                    "message": "PDF uploaded and indexed successfully",
                    "document": {
                        "id": doc_id,
                        "filename": filename,
                        "page_count": page_count
                    }
                }), 200
            except Exception as e:
                # Cleanup on failure
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({"error": f"Error indexing PDF: {str(e)}"}), 500
                
        return jsonify({"error": "Invalid file type. Only PDFs are supported."}), 400

@app.route('/api/documents/<int:doc_id>', methods=['DELETE'])
def api_delete_document(doc_id):
    try:
        # Get filename to remove from disk
        docs = database.get_documents()
        doc_to_delete = None
        for d in docs:
            if d["id"] == doc_id:
                doc_to_delete = d
                break
                
        if doc_to_delete:
            filepath = os.path.join(UPLOAD_FOLDER, doc_to_delete["filename"])
            if os.path.exists(filepath):
                os.remove(filepath)
                
        database.delete_document(doc_id)
        return jsonify({"message": "Document deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Error deleting document: {str(e)}"}), 500

# Chat Sessions
@app.route('/api/chats', methods=['GET', 'POST'])
def api_chats():
    if request.method == 'GET':
        return jsonify(database.get_sessions()), 200
    else:
        session_id = str(uuid.uuid4())
        title = request.json.get('title', 'New Chat')
        database.create_session(session_id, title)
        return jsonify({"id": session_id, "title": title}), 200

@app.route('/api/chats/<session_id>', methods=['GET', 'DELETE', 'PATCH'])
def api_chat_detail(session_id):
    if request.method == 'GET':
        messages = database.get_messages(session_id)
        return jsonify({"messages": messages}), 200
        
    elif request.method == 'PATCH':
        new_title = request.json.get('title')
        if not new_title:
            return jsonify({"error": "Title required"}), 400
        database.rename_session(session_id, new_title)
        return jsonify({"message": "Chat renamed"}), 200
        
    elif request.method == 'DELETE':
        database.delete_session(session_id)
        return jsonify({"message": "Chat deleted"}), 200

# Chat Message Posting (RAG Core logic)
@app.route('/api/chats/<session_id>/messages', methods=['POST'])
def api_chat_message(session_id):
    user_text = request.json.get('text', '').strip()
    active_doc_ids = request.json.get('active_docs', []) # List of document IDs to search in
    
    if not user_text:
        return jsonify({"error": "Message content is required"}), 400
        
    # Get settings
    settings = database.get_settings()
    
    # Save user message to database
    database.add_message(session_id, "user", user_text)
    
    # Retrieve active chunks
    if not active_doc_ids:
        # Fallback: if no active_doc_ids specified, use all documents in DB
        active_doc_ids = [doc["id"] for doc in database.get_documents()]
        
    all_chunks = database.get_all_chunks(active_doc_ids)
    
    citations = []
    response_text = ""
    
    if all_chunks:
        # Run semantic similarity retrieval
        retrieved = rag.retrieve_contexts(user_text, all_chunks, settings, top_k=5)
        
        # Build context prompt
        context_str = ""
        for i, chunk in enumerate(retrieved):
            # Track citations
            citations.append({
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "snippet": chunk["text"]
            })
            context_str += f"[{i+1}] (Source: {chunk['filename']}, Page: {chunk['page_number']}): {chunk['text']}\n\n"
            
        system_instruction = settings.get("system_prompt", "")
        prompt = (
            f"You are a helpful and conversational AI assistant. You have access to some retrieved excerpts from active PDF documents to assist the user.\n"
            f"Use the excerpts to answer the user's input accurately if they are asking about the documents. Cite source pages when doing so.\n"
            f"If the user is saying a greeting, chatting casually, asking for help, or making a request that does not relate to or require the documents, "
            f"chat friendly and respond directly without mentioning the documents or saying 'I cannot find it in the documents'. Be natural and helpful.\n\n"
            f"EXCERPTS:\n{context_str}\n"
            f"USER INPUT: {user_text}\n"
        )
        
        response_text = rag.chat_completion(prompt, settings)
    else:
        # No documents active, direct LLM completion
        system_instruction = settings.get("system_prompt", "") + " (Note: No PDF documents are uploaded or active. Inform the user that you are answering without document context if relevant.)"
        response_text = rag.chat_completion(user_text, settings)
        
    # Save AI response to DB
    database.add_message(session_id, "ai", response_text, citations)
    
    return jsonify({
        "sender": "ai",
        "text": response_text,
        "citations": citations
    }), 200

# Static File serving for PDF.js
@app.route('/uploads/<filename>')
def serve_pdf(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Audio transcription endpoint using ffmpeg conversion for browser recording compatibility
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    import subprocess
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    file = request.files['audio']
    
    # Save the uploaded webm/ogg file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_input:
        file.save(temp_input.name)
        temp_input_path = temp_input.name
        
    temp_output_path = temp_input_path + ".wav"
    
    try:
        # Convert to PCM 16kHz mono WAV using ffmpeg
        subprocess.run([
            'ffmpeg', '-y', '-i', temp_input_path, 
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 
            temp_output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_output_path) as source:
            audio = recognizer.record(source)
            
        text = recognizer.recognize_google(audio)
        return jsonify({"text": text}), 200
        
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "FFmpeg audio conversion failed."}), 500
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech Recognition API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Transcription error: {str(e)}"}), 500
    finally:
        # Clean up temporary files
        if os.path.exists(temp_input_path):
            try:
                os.unlink(temp_input_path)
            except Exception:
                pass
        if os.path.exists(temp_output_path):
            try:
                os.unlink(temp_output_path)
            except Exception:
                pass

# Legacy server-side TTS endpoint for backward compatibility
@app.route('/speak', methods=['POST'])
def speak_text():
    text = request.json.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tf:
        temp_filename = tf.name
        tts.save(temp_filename)
    
    with open(temp_filename, 'rb') as audio_file:
        audio_data = audio_file.read()
    
    os.unlink(temp_filename)
    
    return audio_data, 200, {'Content-Type': 'audio/mpeg'}

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
