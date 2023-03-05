import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale mode
img = cv2.imread('gedung1.jpg', cv2.IMREAD_GRAYSCALE)

# Define thresholds for low, high, and normal intensity levels
low_th = 75
high_th = 175

# Create masks for low, high, and normal intensity levels
low_mask = np.zeros(img.shape, np.uint8)
low_mask[img < low_th] = 255
high_mask = np.zeros(img.shape, np.uint8)
high_mask[img > high_th] = 255
normal_mask = cv2.bitwise_not(cv2.bitwise_or(low_mask, high_mask))

# Calculate histograms for low, high, and normal intensity levels
low_hist = cv2.calcHist([img], [0], low_mask, [256], [0, 256])
high_hist = cv2.calcHist([img], [0], high_mask, [256], [0, 256])
normal_hist = cv2.calcHist([img], [0], normal_mask, [256], [0, 256])

# Show histograms
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.plot(low_hist, color='gray')
plt.xlabel('Intensity level')
plt.ylabel('Frequency')
plt.title(' Kontras Rendah')

plt.subplot(1, 3, 2)
plt.plot(high_hist, color='gray')
plt.xlabel('Intensity level')
plt.ylabel('Frequency')
plt.title(' Kontras Tinggi')

plt.subplot(1, 3, 3)
plt.plot(normal_hist, color='gray')
plt.xlabel('Intensity level')
plt.ylabel('Frequency')
plt.title(' Kontras Normal')

plt.show()
