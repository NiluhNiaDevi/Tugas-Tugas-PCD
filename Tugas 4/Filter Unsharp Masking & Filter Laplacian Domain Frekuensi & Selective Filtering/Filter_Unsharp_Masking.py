import cv2
import numpy as np

# Load image
img = cv2.imread('kamera.jpg', 0)

# Apply Gaussian blur to image
blur = cv2.GaussianBlur(img, (0,0), 8)

# Subtract blurred image from original image to create highpass image
highpass = img - blur

# Apply unsharp masking by adding highpass image to original image
unsharp = img + highpass

# Display filtered image
cv2.imshow('gambar ori', img)
cv2.imshow('Filtered Image', unsharp)
cv2.waitKey(0)
cv2.destroyAllWindows()
