from flask import *
import whisper
import pymupdf4llm as p4
import nltk
import tempfile
import os
from transformers import pipeline
import torch
from pypdf import PdfReader
from google import genai
import base64
from docx import Document
import io

app = Flask(__name__)

client = genai.Client(api_key="AIzaSyBRYOxOU4PpBtxpEr-9a-fVcLJ3KeuuVps")
model = whisper.load_model("base")

def clean(output):
    output = output.replace('##', '')
    output = output.replace('**', '')
    output = output.replace('\n', '<br>')

    return output

def summarize(text):
    summary_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[{
            "parts": [
                {"text": "Give me a summary of the following text and nothing else within a few paragraphs:\n\n" + text}
            ]
        }]
    )
    return summary_response.text

def generate_quiz(text):
    quiz_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[{
            "parts": [
                {"text": text + "\n\nGive me some multiple choice, short answer, fill-in-the-blank, and/or true-or-false quiz questions about the content of the text above and don't say the answers until an answer key at the end of the response. Make sure the correct answer choices are spread out among the options. No prefatory text."}
            ]
        }]
    )
    return quiz_response.text

def extract_text(encoded_img):
    ocr_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[{
            "parts": [
                {"text": "Tell me what this image says and nothing else"},
                {"inline_data": {
                    "mime_type": "image/png", #change to image/png,webp,etc. 
                    "data": encoded_img
                }}
            ]
        }]
    )
    return ocr_response.text

def get_doc_summary(name):

    reader = PdfReader(name)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    summary = summarize(text)
    return clean(summary)

def get_transcript(name):
    return model.transcribe(name)['text']

def get_vid_aud_summary(name):
    transcript = get_transcript(name)
    
    summary = summarize(transcript)
    return clean(summary)

def get_vid_aud_quiz(name):
    transcript = get_transcript(name)

    quiz = generate_quiz(transcript)
    return clean(quiz)

def get_doc_quiz(name):

    reader = PdfReader(name)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    quiz = generate_quiz(text)
    return clean(quiz)

def get_img_txt(name):

    encoded_image_string = base64.b64encode(name.read()).decode("utf-8")

    text = extract_text(encoded_image_string)
    return clean(text)

def download_quiz(quiz):
    doc = Document()
    doc.add_paragraph(quiz)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="my_quiz.docx",
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

def generate_code(request):
    pipe = pipeline(
    "text-generation",
    model="openai/gpt-oss-20b",
    torch_dtype="auto",
    device_map="auto",)
    messages = [
    {"role": "user", "content": "Could you please generate code that answers this prompt? Prompt: " + request},]

    outputs = pipe(
    messages,
    max_new_tokens=256,)
    return outputs[0]["generated_text"]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/summary", methods=["POST", "GET"])
def summary():
    return render_template('summary.html')

@app.route("/summary-result", methods=["POST", "GET"])
def summary_result():
    if (request.method == "POST"):
        file_upload = request.files.get("file")

        if file_upload.mimetype.startswith("audio/") or file_upload.mimetype.startswith("video/"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_upload.filename)[1]) as tmp:
                file_upload.save(tmp.name)
                tmp_path = tmp.name
            try:
                result = get_vid_aud_summary(tmp_path)
            finally:
                os.remove(tmp_path)
        elif file_upload.mimetype == "application/pdf":
            result = get_doc_summary(file_upload)
            
        template = f"<p class='animated' id='summary'>{result}</p>"
        return render_template('summary-result.html') + template
    
    return render_template("summary-result.html")

@app.route("/quiz", methods=["POST", "GET"])
def quiz():
    return render_template('quiz.html')

@app.route("/quiz-result", methods=["POST", "GET"])
def quiz_result():
    if (request.method == "POST"):
        file_upload = request.files.get("file")

        if file_upload.mimetype.startswith("audio/") or file_upload.mimetype.startswith("video/"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_upload.filename)[1]) as tmp:
                file_upload.save(tmp.name)
                tmp_path = tmp.name
            try:
                result = get_vid_aud_quiz(tmp_path)
            finally:
                os.remove(tmp_path)
        elif file_upload.mimetype == "application/pdf":
            result = get_doc_quiz(file_upload)

        template = f"<p class='animated' id='quiz'>{result}</p> \ <button onclick='window.location.href='/quiz-download'>Download Word Document</button>" 

        return render_template('quiz-result.html') + template
    
    return render_template("quiz-result.html")

@app.route("/quiz-download", methods=["POST", "GET"])
def quiz_download():
    return download_quiz("testing")

@app.route("/extract", methods=["POST", "GET"])
def extract():
    return render_template('extract.html')

@app.route("/extract-result", methods=["POST", "GET"])
def extract_result():
    if (request.method == "POST"):
        file_upload = request.files.get("file")
        result = get_img_txt(file_upload)
        template = f"<p class='animated' id='extracted'>{result}</p>"
        return render_template('extract-result.html') + template
    
    return render_template("extract-result.html")

@app.route("/transcribe", methods=["POST", "GET"])
def transcribe():
    return render_template('transcribe.html')

@app.route("/transcribe-result", methods=["POST", "GET"])
def transcribe_result():
    if (request.method == "POST"):
        file_upload = request.files.get("file")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_upload.filename)[1]) as tmp:
            file_upload.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = get_transcript(tmp_path)
            template = f"<p class='animated' id='transcription'>{result}</p>"
            return render_template('transcribe-result.html') + template
        finally:
            os.remove(tmp_path)

    return render_template("transcribe-result.html")

@app.route("/code", methods=["POST", "GET"])
def code():
    result = None
    action = None
    if (request.method == "POST"):
        user_text = request.form.get("text_input")
        action = request.form.get("submit")
        if (action == "summarize"):
            result = user_text
        elif (action == "generate"):
            #result = generate_code(user_text)
            result = user_text
        elif (action == "translate"):
            result = user_text
    return render_template("code.html", result=result, action=action)

if __name__ == "__main__":
    app.run(debug=True)
