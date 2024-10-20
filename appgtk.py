import sys
import os
import threading
import tempfile
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib
import speech_recognition as sr
import requests
import json
from gtts import gTTS
import pygame
import pdfplumber  # Use pdfplumber for PDF handling

class PDFAssistant(Gtk.Window):
    def __init__(self):
        super().__init__(title="PDF Voice Assistant")
        self.set_border_width(10)
        self.set_default_size(600, 400)

        self.pdf_content = ""
        pygame.mixer.init()

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)

        # PDF selection
        hbox = Gtk.Box(spacing=6)
        self.pdf_label = Gtk.Label(label="No PDF selected")
        pdf_button = Gtk.Button(label="Select PDF")
        pdf_button.connect("clicked", self.on_pdf_clicked)
        hbox.pack_start(self.pdf_label, True, True, 0)
        hbox.pack_start(pdf_button, False, False, 0)
        vbox.pack_start(hbox, False, False, 0)

        # Question area
        self.question_area = Gtk.TextView()
        self.question_area.set_wrap_mode(Gtk.WrapMode.WORD)
        question_scroll = Gtk.ScrolledWindow()
        question_scroll.add(self.question_area)
        vbox.pack_start(question_scroll, True, True, 0)

        # Answer area
        self.answer_area = Gtk.TextView()
        self.answer_area.set_wrap_mode(Gtk.WrapMode.WORD)
        self.answer_area.set_editable(False)
        answer_scroll = Gtk.ScrolledWindow()
        answer_scroll.add(self.answer_area)
        vbox.pack_start(answer_scroll, True, True, 0)

        # Buttons
        hbox = Gtk.Box(spacing=6)
        self.record_button = Gtk.Button(label="Record Voice")
        self.record_button.connect("clicked", self.on_record_clicked)
        hbox.pack_start(self.record_button, True, True, 0)

        send_button = Gtk.Button(label="Send Question")
        send_button.connect("clicked", self.on_send_clicked)
        hbox.pack_start(send_button, True, True, 0)

        speak_button = Gtk.Button(label="Speak Answer")
        speak_button.connect("clicked", self.on_speak_clicked)
        hbox.pack_start(speak_button, True, True, 0)

        vbox.pack_start(hbox, False, False, 0)

        self.is_recording = False

    def on_pdf_clicked(self, widget):
        dialog = Gtk.FileChooserDialog(
            title="Please choose a file", parent=self, action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK
        )

        filter_pdf = Gtk.FileFilter()
        filter_pdf.set_name("PDF files")
        filter_pdf.add_mime_type("application/pdf")
        dialog.add_filter(filter_pdf)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            file_path = dialog.get_filename()
            self.pdf_label.set_text(os.path.basename(file_path))
            self.load_pdf(file_path)
        dialog.destroy()

    def load_pdf(self, file_path):
        self.pdf_content = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        self.pdf_content += text
                    else:
                        print("No extractable text on a page.")
                        
            if not self.pdf_content.strip():
                print("No text found in PDF.")
                self.pdf_label.set_text("No text found in PDF.")
            else:
                print(f"PDF loaded. Content length: {len(self.pdf_content)} characters")
        except Exception as e:
            print(f"Error loading PDF: {str(e)}")
            self.pdf_label.set_text("Error loading PDF")

    def on_record_clicked(self, widget):
        if not self.is_recording:
            self.is_recording = True
            self.record_button.set_label("Stop Recording")
            threading.Thread(target=self.record_audio).start()
        else:
            self.is_recording = False
            self.record_button.set_label("Record Voice")

    def record_audio(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio)
                GLib.idle_add(self.set_question_text, text)
            except sr.UnknownValueError:
                GLib.idle_add(self.set_question_text, "Could not understand audio")
            except sr.RequestError as e:
                GLib.idle_add(self.set_question_text, f"Could not request results; {e}")
        GLib.idle_add(self.reset_record_button)

    def set_question_text(self, text):
        self.question_area.get_buffer().set_text(text)

    def reset_record_button(self):
        self.is_recording = False
        self.record_button.set_label("Record Voice")

    def on_send_clicked(self, widget):
        buffer = self.question_area.get_buffer()
        question = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)
        if not question:
            return

        prompt = f"Based on the following PDF content, please answer this question: {question}\n\nPDF Content: {self.pdf_content}"
        threading.Thread(target=self.chat_with_ollama, args=(prompt,)).start()

    def chat_with_ollama(self, prompt):
        try:
            url = "http://localhost:11434/api/generate"
            data = {
                "model": "YOUR_OLLAMA_MODEL",
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
            GLib.idle_add(self.set_answer_text, response_text.strip())
        except requests.exceptions.RequestException as e:
            GLib.idle_add(self.set_answer_text, f"Error communicating with Ollama: {e}")

    def set_answer_text(self, text):
        self.answer_area.get_buffer().set_text(text)

    def on_speak_clicked(self, widget):
        buffer = self.answer_area.get_buffer()
        answer = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)
        if not answer:
            return

        tts = gTTS(text=answer, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tf:
            temp_filename = tf.name
            tts.save(temp_filename)
        
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()

        def cleanup():
            while pygame.mixer.music.get_busy():
                pass  # Wait for the music to finish playing
            os.unlink(temp_filename)

        threading.Thread(target=cleanup).start()

win = PDFAssistant()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()

