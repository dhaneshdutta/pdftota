# pdftota

a rag-based conversational assistant for pdf documents. upload pdfs, ask questions about their content, or just chat — the model dynamically decides when to reference documents and when to respond freely. supports voice input and text-to-speech output.

## what it does

- multi-document rag: upload multiple pdfs, select which ones to include in context, and query across them with source citations
- conversational: doesn't just answer document questions — handles greetings, follow-ups, and general chat naturally
- voice integration: speech-to-text input via microphone recording and text-to-speech playback on responses
- inline pdf viewer: side-by-side pdf rendering with page-level citation linking using pdf.js

## architecture

```
flask (app.py)
├── database.py    sqlite persistence — documents, chunks, embeddings, chat sessions, settings
├── rag.py         text extraction, chunking, embedding generation, similarity search, llm inference
└── templates/
    └── index.html single-page frontend with three-panel layout
```

- embeddings are stored as binary blobs in sqlite — no external vector database required
- retrieval uses cosine similarity on numpy vectors, with a pure python tf-idf fallback when embeddings are unavailable
- audio transcription converts browser webm recordings to wav via ffmpeg before passing to google speech recognition
- tts uses native browser speechsynthesis when available, falls back to server-side gtts

## setup

requires python 3.10+ and ffmpeg installed on the system.

```bash
git clone https://github.com/dhaneshdutta/pdftota.git
cd pdftota
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## usage

```bash
python app.py
```

open `http://127.0.0.1:5000` in your browser.

upload a pdf, check its box in the sidebar to include it in context, and start chatting.

## stack

- backend: flask, pdfplumber, pypdf, speechrecognition, gtts, numpy
- frontend: vanilla html/css/js, pdf.js, marked.js, highlight.js
