import cv2
import numpy as np

# Load image
img = cv2.imread('kamera.jpg')

# Convert to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define color range for blue
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Create mask for blue color range
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Apply selective filtering
blue_channel = cv2.bitwise_and(img, img, mask=mask)
background = cv2.medianBlur(img, 15)
background_masked = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))
selective_filtered = cv2.add(background_masked, blue_channel)

# Display filtered image
cv2.imshow('Original foto', img)
cv2.imshow('Selective Filtering Image', selective_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
