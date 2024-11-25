import cv2
import pytesseract
import re

# Define a function to convert binary strings to letters
def binary_to_text(binary_str):
    try:
        return chr(int(binary_str, 2))  # Convert binary to decimal, then to character
    except ValueError:
        return binary_str  # If conversion fails, return the original string

def text_detection():
    # Load the image
    image_path = "./assets/sample.jpg"
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Perform OCR
    text = pytesseract.image_to_string(thresh)
    
    #binary check
    # Find all binary strings (8 bits assumed for ASCII)
    binary_pattern = r'\b[01]{8}\b'  # Matches binary strings of 8 bits
    binary_strings = re.findall(binary_pattern, text)

    # Convert found binary strings to text
    for binary in binary_strings:
        letter = binary_to_text(binary)
        text = text.replace(binary, letter)
    
    text="Extracted Text:\n"+text
    # Display the text at the bottom of the image
    # Get image dimensions
    height,width = image.shape[:2]
    y0 = height - 50 * (len(text.splitlines()))  # Start text near the bottom
    dy = 30  # Line spacing

    for i, line in enumerate(text.splitlines()):
        y = y0 + i * dy
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, lineType=cv2.LINE_AA)

    # Display the original image with overlaid text
    cv2.imshow("Image with Extracted Text", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()