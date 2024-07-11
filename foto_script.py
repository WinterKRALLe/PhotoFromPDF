"""
PDF Processing Script
Last Updated: 11th of July, 2024

Description:
This script processes PDF documents, extracts birth numbers and detects faces,
and saves cropped face images based on the extracted data.

Libraries Used:
- PyMuPDF (fitz)
- OpenCV (cv2)
- NumPy (np)
- Pillow (Image)
- Regular Expressions (re)
- Tesseract OCR (pytesseract)
"""

import os
import fitz
import cv2
import numpy as np
from PIL import Image
import re
import pytesseract


def extract_birth_number(text):
    """
    Extract the birth number from text.
    Looks for patterns:
    - 10 digits
    - 8 digits followed by 2 letters
    - 9 digits
    """
    pattern = r'\b(\d{10}|\d{8}[A-Za-z]{2}|\d{9})\b'
    matches = re.findall(pattern, text)

    if matches:
        # Return the first match found
        return matches[0]
    return None


def preprocess_image_for_ocr(image_np):
    """
    Preprocesses the image for OCR by converting it to grayscale
    and applying binary thresholding.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def process_pdf(pdf_path, output_directory):
    """
    Processes each page of a PDF to extract the birth number and
    detect faces on the right half of each page. Saves the cropped
    face images with filenames based on the birth number or page number.
    """
    doc = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()

        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Convert to numpy array for further processing
            nparr = np.frombuffer(image_bytes, np.uint8)
            image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Crop the right half of the image
            _, img_w = image_np.shape[:2]
            right_half = image_np[:, img_w//2:]

            # Perform OCR on the right half to get birth number
            text = pytesseract.image_to_string(Image.fromarray(right_half), lang="ces")

            # Try extracting birth number from each line of text
            birth_number = None
            for line in text.splitlines():
                birth_number = extract_birth_number(line)
                if birth_number:
                    break
            print(f"Extracted Birth Number: {birth_number}")  # Debugging: print extracted birth number

            # Detect faces in the right half of the image
            gray_right = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_right, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                # Assume the largest face is the one we want
                x, y, w, h = max(faces, key=lambda item: item[2] * item[3])

                # Calculate dynamic offsets
                face_size_ratio = max(w, h) / min(right_half.shape[:2])
                base_offset = 0.25
                dynamic_offset = max(0.1, base_offset * (1 - face_size_ratio))

                offset_x = int(w * dynamic_offset)
                offset_y = int(h * dynamic_offset * 1.4)

                # Calculate new coordinates with offset
                new_x = max(0, x - offset_x)
                new_y = max(0, y - offset_y)
                new_w = min(right_half.shape[1] - new_x, w + 2*offset_x)
                new_h = min(right_half.shape[0] - new_y, h + 2*offset_y)

                # Crop the face region from right half
                face_image = right_half[new_y:new_y+new_h, new_x:new_x+new_w]

                # Check if face_image is valid before saving
                if face_image.size == 0:
                    print(f"[!] Empty face_image for {pdf_name} page {page_num}. Skipping...")
                    continue

                # Determine the output filename
                if birth_number:
                    output_filename = f"{birth_number}.png"
                else:
                    output_filename = f"{pdf_name}_page{page_num}.png"
                output_filepath = os.path.join(output_directory, output_filename)

                # Save the cropped face image
                cv2.imwrite(output_filepath, face_image)

                print(f"[+] Processed {pdf_name} page {page_num} and saved as {output_filepath}")

    print(f"[!] Finished processing {pdf_name}")


def process_pdf_directory(input_directory, output_directory):
    """
    Processes all PDF files in the input directory, calling process_pdf()
    for each PDF file.
    """
    os.makedirs(output_directory, exist_ok=True)
    pdf_files = [f for f in os.listdir(input_directory) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print("No PDF files found in the input directory.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_directory, pdf_file)
        process_pdf(pdf_path, output_directory)


# Define input and output directories
input_directory = "/mnt/nas/foto/sken-in"
output_directory = "/mnt/nas/foto/sken-out"


# Process the PDFs in the input directory
process_pdf_directory(input_directory, output_directory)
