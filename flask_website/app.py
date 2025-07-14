from flask import *
import os
import tempfile
import whisper
import pymupdf4llm as p4
import nltk
from transformers import pipeline

nltk.download('punkt')
nltk.download('punkt_tab')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

app = Flask(__name__)
app.secret_key = "racetrack_playa"
model = whisper.load_model("base")

"""
GENERAL FUNCTIONS
"""

def summarize(text):
    summarized_chunks = []
    for i in range(0, len(text), 1024):
        summarized_chunks.append(summarizer(text[i: min(i + 1024, len(text))], max_length=130, min_length=30, do_sample=False)[0]['summary_text'])
    ret = ""
    for chunk in summarized_chunks:
        ret += chunk
    return ret

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


"""
WEBSITE FUNCTIONS
"""

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/transcribe-video", methods=["POST", "GET"])
def transcribe_video():
    if (request.method == "POST"):
        file_upload = request.files.get("file")

        if not file_upload:
            return render_template("transcribe.html", summary="Please select a file!")

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_upload.filename)[1]) as tmp:
            file_upload.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = model.transcribe(tmp_path)
            transcription = result['text']

            template = f"<h2 class='content'>Transcription:</h2> \
            <p class='content'>{transcription}</p>"

            return render_template('transcribe.html') + template
        finally:
            os.remove(tmp_path)
    
    return render_template('transcribe.html')

@app.route("/summarize-video", methods=["POST", "GET"])
def summarize_video():
    if (request.method == "POST"):

        file_upload = request.files.get("file")

        if not file_upload:
            return render_template("video.html", summary="Please select a file!")

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_upload.filename)[1]) as tmp:
            file_upload.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = model.transcribe(tmp_path)
            transcription = result['text']

            template = f"<h2 class='content'>Transcription Summary:</h2> \
            <p class='content'>{summarize(transcription)}</p>"

            return render_template('video.html') + template
        finally:
            os.remove(tmp_path)
    
    return render_template('video.html')

@app.route("/summarize-doc", methods=["POST", "GET"])
def summarize_doc():
    if (request.method == "POST"):

        file_upload = request.files.get("file")

        if not file_upload:
            return render_template("document.html", summary="Please select a file!")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_upload.filename)[1]) as tmp:
            file_upload.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            result = get_doc_summary(tmp_path)

            template = f"<h2 class='content'>Document Summary:</h2> \
            <p class='content'>{result}</p>"

            return render_template('document.html') + template
    
        finally:
            os.remove(tmp_path)
    
    return render_template("document.html")

if __name__=='__main__':
    app.run(debug=True)