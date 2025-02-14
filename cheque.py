# USAGE:
# python cheque.py --image example_check.png --reference micr_e13b_reference.png

# Import required packages
from skimage.segmentation import clear_border
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import pytesseract  # For OCR

# Function to extract digits and symbols
def extract_digits_and_symbols(image, charCnts, minW=5, minH=15):
    charIter = charCnts.__iter__()
    rois = []
    locs = []

    while True:
        try:
            c = next(charIter)
            (cX, cY, cW, cH) = cv2.boundingRect(c)
            roi = None

            if cW >= minW and cH >= minH:
                roi = image[cY:cY + cH, cX:cX + cW]
                rois.append(roi)
                locs.append((cX, cY, cX + cW, cY + cH))
            else:
                parts = [c, next(charIter), next(charIter)]
                (sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf, -np.inf)

                for p in parts:
                    (pX, pY, pW, pH) = cv2.boundingRect(p)
                    sXA = min(sXA, pX)
                    sYA = min(sYA, pY)
                    sXB = max(sXB, pX + pW)
                    sYB = max(sYB, pY + pH)

                roi = image[sYA:sYB, sXA:sXB]
                rois.append(roi)
                locs.append((sXA, sYA, sXB, sYB))

        except StopIteration:
            break

    return (rois, locs)

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input check image")
ap.add_argument("-r", "--reference", required=True, help="Path to reference MICR E-13B font")
args = vars(ap.parse_args())

# List of MICR character names
charNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "T", "U", "A", "D"]

# Load the reference MICR image
ref = cv2.imread(args["reference"])

# Check if the reference image was loaded successfully
if ref is None:
    print(f"❌ Error: Could not load reference image '{args['reference']}'")
    exit(1)

# Convert to grayscale and resize
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = imutils.resize(ref, width=400)

# Display the reference image before processing
cv2.imshow("Reference Image", ref)
cv2.waitKey(1000)  # Auto-close after 1 sec
cv2.destroyAllWindows()

# Improve contrast before thresholding
ref = cv2.convertScaleAbs(ref, alpha=2, beta=10)

# Apply Gaussian Blur to remove noise
ref = cv2.GaussianBlur(ref, (5, 5), 0)

# Apply thresholding
ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Display the processed reference image
cv2.imshow("Thresholded Reference", ref)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# Find contours in the reference image
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Ensure compatibility with OpenCV versions
refCnts = refCnts[0] if len(refCnts) == 2 else refCnts[1]

# Check if any contours were found
if len(refCnts) == 0:
    print("❌ Error: No contours found in the reference image.")
    exit(1)

# Sort contours from left to right
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

# Extract digits and symbols
(refROIs, refLocs) = extract_digits_and_symbols(ref, refCnts, minW=10, minH=20)
chars = {}

# Recognize MICR characters using OCR
recognized_text = ""
for (name, roi, loc) in zip(charNames, refROIs, refLocs):
    roi = cv2.resize(roi, (36, 36))  # Resize for uniform OCR processing
    
    # Apply OCR to recognize the character
    text = pytesseract.image_to_string(roi, config="--psm 10 --oem 3", lang="eng").strip()
    
    # Append recognized text
    recognized_text += text

print("✅ Extracted MICR Text:", recognized_text)

# Load the input check image
image = cv2.imread(args["image"])

# Check if the check image was loaded successfully
if image is None:
    print(f"❌ Error: Could not load check image '{args['image']}'")
    exit(1)

# Convert to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = imutils.resize(image, width=600)

# Improve contrast
image = cv2.convertScaleAbs(image, alpha=2, beta=10)

# Apply Gaussian blur
image = cv2.GaussianBlur(image, (5, 5), 0)

# Apply thresholding
image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Display the processed check image
cv2.imshow("Processed Check Image", image)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# Apply OCR to extract text from the cheque
cheque_text = pytesseract.image_to_string(image, config="--psm 6 --oem 3", lang="eng").strip()

# Print the extracted text from the cheque
print("✅ Extracted Cheque Text:", cheque_text)

print("🎯 Processing complete!")
