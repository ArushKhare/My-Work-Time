from PIL import Image #Pillow
import pytesseract
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import DonutProcessor
import torch
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select a PDF File or Image:",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png"), ("PDF files", "*.pdf")]
)

if file_path:
    print(f"Selected file: {file_path}")
    file_extension = os.path.splitext(file_path)[1].lower()
    if (file_extension == ".pdf"):
        print("pdf file")
    else: #must be an image file
        # #pytesseract: generally worse for handwriting
        # text = pytesseract.image_to_string(Image.open(file_path).convert('RGB'))

        # easyOCR - better at symbols, don't need to separate into lines
        # reader = easyocr.Reader(['en'])
        # text = reader.readtext(file_path, detail = 0)

        #trOCR - better handwriting recognition but needs individual lines. Also works on the good cursive writing!
        #Image chopping not working yet; use trOCR on one-line examples
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

        #image chopping: currently only works on multiple lines if the lines are spaced pretty far apart
        text = ""
        image = cv2.imread(file_path)
        height, width, channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lineRects = sorted([cv2.boundingRect(c) for c in contours], key = lambda b: b[1])

        for i, (x, y, w, h) in enumerate(lineRects):
            line = image[y:y+h, x:x+w]
            if line.shape[2] != 3 or h < 2: continue
            line = cv2.cvtColor(line, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(line)
            #pil_img.show() #opens pop-up images of the individual lines
            pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            text += processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            text += " "

        print("\nOCR Recognized this text:")
        print(text)
else:
    print("Not valid file")