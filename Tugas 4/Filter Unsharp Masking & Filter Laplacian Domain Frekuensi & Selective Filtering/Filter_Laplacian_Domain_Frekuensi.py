import cv2
import numpy as np
import scipy.fftpack as fftpack

# Load image
img = cv2.imread('kamera.jpg', 0)

# Apply 2D Fourier transform to image
f = fftpack.fft2(img)

# Create Laplacian filter
rows, cols = img.shape
crow, ccol = rows/2, cols/2
L = 10 # Laplacian gain
D = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        D[i, j] = np.sqrt((i - crow)**2 + (j - ccol)**2)
H = (L * (D**2))

# Apply Laplacian filter in frequency domain
f_shifted = fftpack.fftshift(f)
f_filtered = f_shifted * H
f_inv = fftpack.ifftshift(f_filtered)
img_filtered = np.real(fftpack.ifft2(f_inv))

# Display filtered image
cv2.imshow('gambar asli', img)
cv2.imshow('Filtered Image', img_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
