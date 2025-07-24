import base64
from pypdf import PdfReader #pip install pypdf
from google import genai #pip install -q -U google-genai

client = genai.Client(api_key="INSERT-YOUR-API-KEY")

#comment out any section of testing as needed

# #OCR TESTING------------------------------------------------------------------
img_path = r"insert/local/image/path/here"

#in order for gemini to read the image, it has to go through base64 encoding
with open(img_path, "rb") as image_file:
    encoded_image_string = base64.b64encode(image_file.read()).decode("utf-8")

ocr_response = client.models.generate_content(
    model="gemini-2.0-flash", #can try other gemini models
    contents=[{
        "parts": [
            {"text": "Tell me what this image says and nothing else"},
            {"inline_data": {
                "mime_type": "image/jpeg", #change to image/png,webp,etc. as needed
                "data": encoded_image_string
            }}
        ]
    }]
)
print(ocr_response.text)

#SUMMARIZATION TESTING----------------------------------------------------------
text_path = r"insert/local/pdf/file/path/here"

#can't read the file directly like images, so extract text using pypdf first
reader = PdfReader(text_path) #the user-uploaded pdf file
text = ""
for page in reader.pages:
    text += page.extract_text()

summary_response = client.models.generate_content(
    model="gemini-2.0-flash", #can try other gemini models
    contents=[{
        "parts": [
            {"text": "Give me a summary of the following text and nothing else within a few short paragraphs:\n\n" + text}
        ]
    }]
    
)
print(summary_response.text)

#QUIZ QUESTIONS TESTING----------------------------------------------------------
content_path = r"insert/local/file/path/here"

reader = PdfReader(content_path) #the user-uploaded pdf file
content = ""
for page in reader.pages:
    content += page.extract_text()

quiz_response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[{
        "parts": [
            {"text": content + "\n\nGive me some multiple choice, short answer, fill-in-the-blank, and/or true-or-false quiz questions about the content of the text above and don't say the answers until an answer key at the end of the response. Make sure the correct answer choices are spread out among the options. No prefatory text."}
        ]
    }]
)
print(quiz_response.text)
