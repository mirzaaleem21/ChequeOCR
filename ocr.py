import cv2
import pytesseract
import numpy as np
import argparse
import imutils
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border

# Function to preprocess cheque image for better OCR
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Apply Median Blur to reduce noise
    gray = cv2.medianBlur(gray, 3)

    # Adaptive Thresholding to binarize text
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4
    )

    return binary

# Function to extract text using OCR
def extract_text(image):
    text = pytesseract.image_to_string(
        image, config="--oem 3 --psm 4", lang="eng"
    ).strip()
    return text

# Argument Parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input cheque image")
args = vars(ap.parse_args())

# Load cheque image
image = cv2.imread(args["image"])
if image is None:
    print(f"❌ Error: Could not load cheque image '{args['image']}'")
    exit(1)

# Preprocess image for OCR
processed_image = preprocess_image(image)

# Extract cheque text
cheque_text = extract_text(processed_image)

# Display Preprocessed Image for Debugging
plt.figure(figsize=(10, 5))
plt.imshow(processed_image, cmap='gray')
plt.title("Preprocessed Cheque Image")
plt.axis("off")
plt.show()

# Print extracted text
print("✅ Extracted Cheque Text:")
print(cheque_text)
print("🎯 Processing complete!")
