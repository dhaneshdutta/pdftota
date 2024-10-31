from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import tempfile
import os
import pdfplumber  # Switch to pdfplumber for PDF handling
import requests
import json
from gtts import gTTS

app = Flask(__name__)

pdf_content = ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global pdf_content
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.pdf'):
        pdf_content = ""
        # Using pdfplumber to extract text
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pdf_content += text
                    else:
                        print("No extractable text on a page.")
        
            if not pdf_content.strip():
                return jsonify({"error": "No text found in PDF"}), 400
            
            print(f"PDF uploaded. Content length: {len(pdf_content)} characters")
            return jsonify({"message": "PDF uploaded successfully", "content": pdf_content[:100]}), 200
        except Exception as e:
            return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500
            
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    file = request.files['audio']
    recognizer = sr.Recognizer()
    audioFile = sr.AudioFile(file)
    with audioFile as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return jsonify({"text": text}), 200
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Could not request results; {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    global pdf_content
    question = request.json.get('question', '')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    prompt = f"Based on the following PDF content, please answer this question: {question}\n\nPDF Content: {pdf_content}"
    
    try:
        url = "http://localhost:11434/api/generate"
        data = {
            "model": "pdftota",
            "prompt": prompt
        }
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        response_text = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                response_json = json.loads(decoded_line)
                if 'response' in response_json:
                    response_text += response_json['response']
        return jsonify({"answer": response_text.strip()}), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error communicating with Ollama: {str(e)}"}), 500

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
    app.run(debug=True)

