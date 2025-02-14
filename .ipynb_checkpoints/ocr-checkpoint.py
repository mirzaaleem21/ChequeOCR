from PIL import Image
import pytesseract
import numpy as np

filename = 'image.jpeg'
img1 = np.array(Image.open(filename))
text = pytesseract.image_to_string(img1)
