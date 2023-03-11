import cv2
import numpy as np
import matplotlib.pyplot as plt

# membaca gambar dalam skala keabuan
img = cv2.imread('buah.jpeg', cv2.IMREAD_GRAYSCALE)

# menghitung DFT dari gambar
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# menampilkan gambar asli dan magnitude spectrum dari DFT
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Frekuensi Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
