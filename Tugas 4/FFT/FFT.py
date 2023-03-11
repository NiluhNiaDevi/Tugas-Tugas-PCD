import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image in grayscale
img = cv2.imread('nia.jpeg', cv2.IMREAD_GRAYSCALE)

# Calculate FFT
fft_img = np.fft.fft2(img)
fft_img = np.fft.fftshift(fft_img)

# Calculate magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(fft_img))

# Display original and magnitude spectrum
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('frekuensi Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
