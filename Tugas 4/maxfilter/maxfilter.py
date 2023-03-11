import cv2
import numpy as np

# Load gambar
img = cv2.imread('buah.jpeg', 0)

# Definisikan ukuran kernel
kernel_size = 4

# Buat kernel max filter
kernel_max = np.ones((kernel_size, kernel_size), np.uint8)

# Aplikasikan max filter
img_max = cv2.dilate(img, kernel_max, iterations=1)

# Tampilkan gambar asli dan hasil max filter
cv2.imshow('Gambar Asli', img)
cv2.imshow('Hasil Max Filter', img_max)
cv2.waitKey(0)
cv2.destroyAllWindows()
