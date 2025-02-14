import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image_path = "check.jpg"

img = cv2.imread(image_path)
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Increase contrast
gray = cv2.convertScaleAbs(gray, alpha=2, beta=5)

# Denoise using Non-Local Means Denoising
gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

# Adaptive Thresholding (less aggressive)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Show the preprocessed image
plt.imshow(thresh, cmap='gray')
plt.axis("off")
plt.show()

# Perform OCR with improved settings
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(thresh, config=custom_config, lang='eng')

# Print extracted text
print("Extracted Text:", text)
