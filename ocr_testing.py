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

        #---------------------------------------------------------------------#
        #trOCR - better handwriting recognition but needs individual lines. Also partially works on cursive
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

        text = ""
        image = cv2.imread(file_path)
        height, width, channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        #method 1:dilation/contours: currently only works on multiple lines if the lines are spaced pretty far apart
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        # dilated = cv2.dilate(thresh, kernel, iterations=2)
        # contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # lineRects = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: (b[1] // 10, b[0]))

        # for i, (x, y, w, h) in enumerate(lineRects):
        #     line = image[y:y+h, x:x+w]
        #     if line.shape[2] != 3 or h < 2: continue
        #     line = cv2.cvtColor(line, cv2.COLOR_BGR2RGB)
        #     pil_img = Image.fromarray(line)
        #     # pil_img.show() #opens pop-up images of the individual lines
        #     pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
        #     generated_ids = model.generate(pixel_values)
        #     text += processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #     text += " "

        #method 2: horizontal projection: works better; still some small issues
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        thresh = cv2.subtract(thresh, detected_lines)
        horizontal_sum = np.sum(thresh, axis=1)
        line_bounds = []
        in_line = False
        threshold = width * 10
        min_height = 10
        for y, val in enumerate(horizontal_sum):
            if val > threshold and not in_line:
                start = y
                in_line = True
            elif val <= threshold and in_line:
                end = y
                in_line = False
                if end - start > 10:
                    line_bounds.append((start, end))

        text = ""
        padding = 7
        for start, end in line_bounds:
            start = max(0, start - padding)
            end = min(thresh.shape[0], end + padding)
            line_img = image[start:end, :]
            pil_img = Image.fromarray(line_img)
            with torch.no_grad():
                pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values, do_sample = False)
                decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                text += decoded.strip() + "\n"

        print("\nOCR Recognized this text:")
        print(text)
else:
    print("Not valid file")
