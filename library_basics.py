import re
import tkinter as tk
from tkinter import filedialog
from summarizer import Summarizer #pip install bert-extractive-summarizer; might have to change your numpy version
from pypdf import PdfReader #pip install pypdf

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select a PDF File:",
    filetypes=[("PDF files", "*.pdf")]
)

if file_path:
    print(f"Selected file: {file_path}")
    reader = PdfReader(file_path) #the user-uploaded pdf file
    text = ""
    for page in reader.pages:
        text += page.extract_text() #this is the core function from pypdf

    # reader.pages is an array of the file's pages, so it can be indexed to get
    # certain page ranges and ignore pages like works cited

    # uncomment below to see the text extracted by pypdf before summarization
    # print("\nExtracted text:\n")
    # print(text)

    model = Summarizer() #can take in parameters to adjust model a bit
    summary = model(text, num_sentences = 10) #can take in parameters to adjust summary length/settings

    #the next three lines clean up summary text formatting; otherwise summary has many unnecessary line breaks
    cleaned_summary = summary.replace("-\n", "") # Fix hyphenated words
    cleaned_summary = re.sub(r"(?<!\n)\n(?!\n)", " ", summary)  # Turn line breaks into spaces
    cleaned_summary = re.sub(r"\n{2,}", "\n\n", cleaned_summary)  # Keep paragraph breaks

    print("\n=== Summary ===\n")
    print(cleaned_summary)
else:
    print("No valid file selected.")