import cv2
import numpy as np

# Load image
img = cv2.imread('buah.jpeg', cv2.IMREAD_GRAYSCALE)

# Define kernel size
kernel_size = 3

# Apply minimum filter
min_filtered_img = cv2.erode(img, np.ones((kernel_size,kernel_size),np.uint8))

# Show output image
cv2.imshow('Original Image', img)
cv2.imshow('Minimum Filtered Image', min_filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
