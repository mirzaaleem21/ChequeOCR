# USAGE: python cheque.py --image check.jpg

import cv2
import pytesseract
import numpy as np
import argparse
import imutils
import re

# Function to extract the cheque issuer's name (Top-Left Region)
def extract_issuer_name(image):
    h, w = image.shape[:2]
    name_roi = image[int(h * 0.08):int(h * 0.20), int(w * 0.05):int(w * 0.50)]

    gray = cv2.cvtColor(name_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=3, beta=15)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    cv2.imshow("Issuer Name ROI", thresh)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

    raw_text = pytesseract.image_to_string(thresh, config="--psm 7 --oem 3", lang="eng").strip()
    name_match = re.findall(r'[A-Za-z\s]+', raw_text)
    return name_match[0].strip() if name_match else "❌ Not Detected"

# Function to extract the receiver name (Handwritten beside "Pay to the Order of")
def extract_receiver_name(image):
    h, w = image.shape[:2]

    # Crop the region where the receiver's name is located
    receiver_roi = image[int(h * 0.35):int(h * 0.47), int(w * 0.35):int(w * 0.70)]

    # Convert to grayscale
    gray = cv2.cvtColor(receiver_roi, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=3, beta=15)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )

    # Debugging: Show the extracted receiver name region
    cv2.imshow("Receiver ROI", thresh)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

    # OCR with PSM 7 (single-line recognition)
    raw_text = pytesseract.image_to_string(thresh, config="--psm 7 --oem 3", lang="eng").strip()

    # Filter out unwanted symbols/numbers using regex
    name_match = re.findall(r'[A-Za-z\s]+', raw_text)
    return name_match[0].strip() if name_match else "❌ Not Detected"


# Function to extract and format the amount correctly
def extract_amount(image):
    h, w = image.shape[:2]

    # Crop the region where the amount is located
    amount_roi = image[int(h * 0.40):int(h * 0.50), int(w * 0.60):int(w * 0.95)]

    # Convert to grayscale
    gray = cv2.cvtColor(amount_roi, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=3, beta=15)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Debugging: Show extracted amount region
    cv2.imshow("Amount ROI", thresh)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

    # OCR with PSM 6 (structured number detection)
    raw_text = pytesseract.image_to_string(thresh, config="--psm 6 --oem 3", lang="eng").strip()

    # Extract numbers and replace "," with "."
    amount_match = re.search(r'([\d,]+\.\d{2})', raw_text.replace(",", "."))
    return f"${amount_match.group(1)}" if amount_match else "❌ Not Detected"


# Argument Parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input check image")
args = vars(ap.parse_args())

# Load the cheque image
image = cv2.imread(args["image"])
if image is None:
    print(f"❌ Error: Could not load cheque image '{args['image']}'")
    exit(1)

# Preprocess the image (resize for better readability)
image = imutils.resize(image, width=1000)

# Extract Details
issuer_name = extract_issuer_name(image)
receiver_name = extract_receiver_name(image)
cheque_amount = extract_amount(image)

# Display Results
print(f"✅ Cheque Issuer Name: {issuer_name}")
print(f"✅ Cheque Receiver Name: {receiver_name}")
print(f"✅ Extracted Amount: {cheque_amount}")
print("🎯 Processing complete!")
