import os
import cv2
import pytesseract
from pdf2image import convert_from_path

def pdf_to_images(pdf_path, output_folder, dpi=300):
    """
    Convert each page of a PDF to an image.
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    pages = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []
    
    for i, page in enumerate(pages):
        image_path = os.path.join(output_folder, f"output_page_{i}.png")
        page.save(image_path, 'PNG')
        image_paths.append(image_path)
        print(f"Saved image: {image_path}")
    
    return image_paths

def ocr_image(image_path):
    """
    Perform OCR on a single image.
    """
    # Read image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image at {image_path}")
    
    # Convert to grayscale for better OCR performance
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Optional: You can add more pre-processing steps here (e.g., thresholding)
    
    # Extract text using pytesseract
    text = pytesseract.image_to_string(gray)
    return text

def extract_text_from_pdf(pdf_path, output_folder):
    """
    Process a PDF file: convert to images and extract text via OCR.
    """
    image_paths = pdf_to_images(pdf_path, output_folder)
    full_text = ""
    
    for idx, image_path in enumerate(image_paths):
        text = ocr_image(image_path)
        full_text += f"\n--- Page {idx} ({image_path}) ---\n{text}\n"
        print(f"OCR complete for: {image_path}")
    
    return full_text

def save_text(text, filename):
    """
    Save the extracted text to a file.
    """
    with open(filename, "w") as f:
        f.write(text)
    print(f"Extracted text saved to: {filename}")

def main():
    # Specify your PDF file path and output folder for images
    pdf_path = "sample_document.pdf"  # Replace with your PDF file
    output_folder = "output_images"
    
    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(pdf_path, output_folder)
    
    # Save the extracted text to a file
    save_text(extracted_text, "extracted_text.txt")
    
    # Further steps could include table extraction, dataset construction,
    # fine-tuning a model, and setting up an API for inference.
    print("Processing complete.")

if __name__ == "__main__":
    main()
