
# PDFtota 📄🤖

**PDFtota** is a PDF-based voice assistant that allows users to upload a PDF, ask questions about its content, and receive responses from an LLM (via an API). It also supports voice recording for input and text-to-speech (TTS) to speak out the AI's responses.

## Features ✨

- **📂 PDF Uploading:** Upload a PDF file and extract its content for analysis.
- **🎤 Voice Input:** Record your question via microphone and transcribe it into text.
- **💬 Text Input:** Ask questions directly through a chat-like interface.
- **⚡ AI-Powered Responses:** Get answers based on the PDF's content using a local LLM model.
- **🔊 Text-to-Speech (TTS):** Listen to the AI's responses via a button beside each answer.
- **🌑 Dark-Themed, Fullscreen Chat UI:** A modern chat-like interface styled with minimalistic, grey tones.

## Technologies Used 💻

- **Frontend:**
  - HTML, CSS, JavaScript
  - Bootstrap icons
  - Fullscreen, chat-like UI similar to ChatGPT's interface.
- **Backend:**
  - Python (Flask)
  - SpeechRecognition for voice input
  - PyMuPDF (fitz) for PDF text extraction
  - gTTS for text-to-speech
  - Local LLM integration via an API

## Installation 🛠️

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/PDFtota.git
   cd PDFtota
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.x installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask Server:**
   ```bash
   python app.py
   ```

4. **Access the Application:**
   Open your web browser and go to `http://127.0.0.1:5000/`.

## Usage 🚀

- **📥 Upload a PDF:** Use the PDF icon next to the chat box to upload a PDF.
- **💬 Ask a Question:** Type your question or record your voice, then press the **send** button to get a response.
- **🔊 Speak the Response:** After a response is received, you will see a button next to the response bubble that allows you to hear the answer using TTS.

## Project Structure 🗂️

```bash
.
├── static/
│   ├── css/
│   ├── js/
├── templates/
│   ├── index.html
├── app.py
├── requirements.txt
└── README.md
```

## License ⚖️

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute, suggest improvements, or report issues! 😊
