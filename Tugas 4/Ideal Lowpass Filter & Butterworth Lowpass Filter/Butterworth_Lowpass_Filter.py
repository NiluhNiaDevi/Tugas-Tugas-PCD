import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load gambar
img = cv2.imread('pemandangan.jpg', 0)

# Lakukan transformasi Fourier pada gambar
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Buat mask untuk lowpass filter
baris, kolom = img.shape
pusat_baris, pusat_kolom = int(baris / 2), int(kolom / 2)
mask = np.zeros((baris, kolom), np.uint8)

# Buat Butterworth Lowpass Filter
radius_butterworth = 30
n = 2
for i in range(baris):
    for j in range(kolom):
        jarak = np.sqrt((i - pusat_baris) ** 2 + (j - pusat_kolom) ** 2)
        mask[i, j] = 1 / (1 + (jarak / radius_butterworth) ** (2 * n))

# Terapkan Butterworth Lowpass Filter
fshift_filtered = fshift * mask
f_filtered = np.fft.ifftshift(fshift_filtered)
img_filtered = np.fft.ifft2(f_filtered)
img_filtered = np.abs(img_filtered)

# Plot gambar dan spektrum Fourier
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Gambar Asli')
axs[0, 1].imshow(np.log(1 + np.abs(fshift)), cmap='gray')
axs[0, 1].set_title('Spektrum Fourier')
axs[1, 0].imshow(mask, cmap='gray')
axs[1, 0].set_title('Butterworth Lowpass Filter')
axs[1, 1].imshow(img_filtered, cmap='gray')
axs[1, 1].set_title('Gambar Setelah Filtering')
plt.show()
