from flask import *
import os
import tempfile
import whisper
import pymupdf4llm as p4
import nltk
from transformers import pipeline
from pypdf import PdfReader
from google import genai
import base64

app = Flask(__name__)

client = genai.Client(api_key="INSERT-YOUR-KEY")

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

def get_doc_summary(name):
    reader = p4.LlamaMarkdownReader()
    data = reader.load_data(os.path.abspath(name))

    text = ""
    for i in range(len(data)):
        text += data[i].text_resource.text
    lines = text.split("\n")
    for i in range(len(lines)):
        lines[i] = lines[i].replace("##", "")
        lines[i] = lines[i].replace("**", "")
        lines[i] = lines[i].replace("\u200b", "-->")
        if (lines[i] != "" and lines[i][0] == "#"):
            lines[i] = lines[i][1::]
        lines[i] = lines[i].strip().rstrip()
    
    ret = []
    for i in range(len(lines)):
        if (lines[i].lower() in ["references", "works cited"]):
            break
        if (lines[i] != ''):
            ret.append(lines[i])
    lines = ret

    t = ""
    for line in lines:
        t += line + "\n"

    return summarize(t)

def get_doc_quiz(name):
    reader = p4.LlamaMarkdownReader()
    data = reader.load_data(os.path.abspath(name))

    text = ""
    for i in range(len(data)):
        text += data[i].text_resource.text
    lines = text.split("\n")
    for i in range(len(lines)):
        lines[i] = lines[i].replace("##", "")
        lines[i] = lines[i].replace("**", "")
        lines[i] = lines[i].replace("\u200b", "-->")
        if (lines[i] != "" and lines[i][0] == "#"):
            lines[i] = lines[i][1::]
        lines[i] = lines[i].strip().rstrip()
    
    ret = []
    for i in range(len(lines)):
        if (lines[i].lower() in ["references", "works cited"]):
            break
        if (lines[i] != ''):
            ret.append(lines[i])
    lines = ret

    t = ""
    for line in lines:
        t += line + "\n"

    return generate_quiz(t)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/summary", methods=["POST", "GET"])
def summary():
    return render_template('summary.html')

@app.route("/summary-doc", methods=["POST", "GET"])
def summary_doc():
    if (request.method == "POST"):

        file_upload = request.files.get("file")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_upload.filename)[1]) as tmp:
            file_upload.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            result = get_doc_summary(tmp_path).replace('\n', '<br>')

            template = f"<p class='animated' id='summary'>{result}</p>"

            return render_template('summary-doc.html') + template
    
        finally:
            os.remove(tmp_path)
    
    return render_template("summary-doc.html")

@app.route("/quiz", methods=["POST", "GET"])
def quiz():
    return render_template('quiz.html')

@app.route("/quiz-doc", methods=["POST", "GET"])
def quiz_doc():
    if (request.method == "POST"):

        file_upload = request.files.get("file")

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_upload.filename)[1]) as tmp:
            file_upload.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            result = get_doc_quiz(tmp_path).replace('\n', '<br>')
            result = result.replace('**', '')

            template = f"<p class='animated' id='quiz'>{result}</p>"

            return render_template('quiz-doc.html') + template
    
        finally:
            os.remove(tmp_path)

    return render_template("quiz-doc.html")

if __name__ == "__main__":
    app.run(debug=True)