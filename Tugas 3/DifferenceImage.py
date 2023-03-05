import cv2
import numpy as np

# Load images
img1 = cv2.imread('lena1.jpg')
img2 = cv2.imread('lena2.jpg')

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization
equalized1 = cv2.equalizeHist(gray1)
equalized2 = cv2.equalizeHist(gray2)

# Calculate difference image
diff = cv2.absdiff(equalized1, equalized2)

# Display images
cv2.imshow('Original Image 1', gray1)
cv2.imshow('Original Image 2', gray2)
cv2.imshow('Equalized Image 1', equalized1)
cv2.imshow('Equalized Image 2', equalized2)
cv2.imshow('Difference Image', diff)

cv2.waitKey(0)
cv2.destroyAllWindows()
