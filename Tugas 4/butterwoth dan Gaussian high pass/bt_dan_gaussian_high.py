import cv2
import numpy as np
from scipy import ndimage

def butterworth_highpass_filter(image, cutoff, n):
    # Menghitung dimensi gambar
    rows, cols = image.shape[:2]

    # Membuat meshgrid
    x, y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))

    # Menghitung jarak dari setiap titik ke pusat
    distance = np.sqrt(x ** 2 + y ** 2)

    # Membuat highpass filter Butterworth
    filter = 1 / (1 + (distance / cutoff) ** (2 * n))

    # Menerapkan filter ke gambar
    filtered_image = np.fft.ifft2(np.fft.fft2(image) * filter).real

    return filtered_image

def gaussian_highpass_filter(image, sigma):
    # Menerapkan filter Gaussian pada gambar
    blurred_image = ndimage.gaussian_filter(image, sigma)

    # Mengurangi gambar asli dengan gambar yang telah di-blur
    filtered_image = image - blurred_image

    return filtered_image

def combined_highpass_filter(image, butterworth_cutoff, butterworth_n, gaussian_sigma):
    # Menerapkan Butterworth Highpass Filter
    butterworth_filtered = butterworth_highpass_filter(image, butterworth_cutoff, butterworth_n)

    # Menerapkan Gaussian Highpass Filter
    gaussian_filtered = gaussian_highpass_filter(image, gaussian_sigma)

    # Menambahkan kedua hasil filter
    combined_filtered = butterworth_filtered + gaussian_filtered

    # Normalisasi hasil
    combined_filtered = cv2.normalize(combined_filtered, None, 0, 255, cv2.NORM_MINMAX)

    return combined_filtered.astype(np.uint8)

# Contoh penggunaan
image = cv2.imread('panda.jpg', cv2.IMREAD_GRAYSCALE)
filtered_image = combined_highpass_filter(image, 30, 2, 2)
cv2.imshow('original', image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
