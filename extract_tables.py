import os
import sys
import cv2
import csv
import pytesseract

# For PDF table extraction
import camelot

def extract_tables_from_pdf(pdf_path, output_folder, flavor="stream"):
    """
    Extract tables from a PDF file using Camelot and save each table as a CSV file.
    """
    os.makedirs(output_folder, exist_ok=True)
    tables = camelot.read_pdf(pdf_path, pages='all', flavor=flavor)
    print(f"Found {len(tables)} table(s) in {pdf_path}")
    
    for i, table in enumerate(tables):
        csv_path = os.path.join(output_folder, f'table_{i}.csv')
        table.to_csv(csv_path)
        print(f"Table {i} saved as CSV to {csv_path}")
        print("Extracted Table DataFrame:")
        print(table.df)
    
    return tables

def extract_table_from_image(image_path, output_csv):
    """
    Extract a table from an image by detecting grid lines using morphological operations,
    then run OCR on each detected cell to build a CSV.
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Unable to read image: {image_path}")
        return
    
    # Convert to grayscale and invert
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    
    # Adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, -2)
    
    # Detect horizontal lines
    horizontal = thresh.copy()
    cols = horizontal.shape[1]
    horizontal_size = max(1, cols // 30)
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)
    
    # Detect vertical lines
    vertical = thresh.copy()
    rows = vertical.shape[0]
    vertical_size = max(1, rows // 30)
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)
    
    # Combine horizontal and vertical lines to get the grid
    grid = cv2.add(horizontal, vertical)
    
    # Find contours in the grid image
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and collect cell bounding boxes
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 50 or h < 20:  # filter out noise; adjust thresholds as needed
            continue
        cells.append((x, y, w, h))
    
    if not cells:
        print("No table cells detected.")
        return

    # Sort cells top-to-bottom then left-to-right
    cells = sorted(cells, key=lambda b: (b[1], b[0]))
    
    # Group cells into rows based on y-coordinate proximity
    rows_list = []
    current_row = []
    current_y = -1
    for cell in cells:
        x, y, w, h = cell
        if current_y == -1 or abs(y - current_y) < 10:
            current_row.append(cell)
            current_y = y
        else:
            rows_list.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [cell]
            current_y = y
    if current_row:
        rows_list.append(sorted(current_row, key=lambda b: b[0]))
    
    # OCR each cell to extract text
    table_data = []
    for row in rows_list:
        row_data = []
        for cell in row:
            x, y, w, h = cell
            cell_img = img[y:y+h, x:x+w]
            # Use --psm 7 for a single line of text
            text = pytesseract.image_to_string(cell_img, config='--psm 7').strip()
            row_data.append(text)
        table_data.append(row_data)
    
    # Save the table data to CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in table_data:
            writer.writerow(row)
    
    print(f"Extracted table saved to {output_csv}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file>")
        return
    
    input_file = sys.argv[1]
    ext = os.path.splitext(input_file)[1].lower()
    
    if ext == ".pdf":
        output_folder = "extracted_tables_pdf"
        extract_tables_from_pdf(input_file, output_folder, flavor="stream")
    elif ext in [".png", ".jpg", ".jpeg"]:
        output_csv = "extracted_table_image.csv"
        extract_table_from_image(input_file, output_csv)
    else:
        print("Unsupported file type. Please provide a PDF or an image file.")

if __name__ == "__main__":
    main()
