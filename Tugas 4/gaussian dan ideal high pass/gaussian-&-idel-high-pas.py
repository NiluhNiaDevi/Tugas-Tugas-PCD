import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load gambar
img = cv2.imread('gedung.jpg', 0)

# Lakukan transformasi Fourier pada gambar
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Buat mask untuk lowpass filter Gaussian
baris, kolom = img.shape
pusat_baris, pusat_kolom = int(baris / 2), int(kolom / 2)
sigma = 50
mask = np.zeros((baris, kolom), np.uint8)
for i in range(baris):
    for j in range(kolom):
        jarak = np.sqrt((i - pusat_baris) ** 2 + (j - pusat_kolom) ** 2)
        mask[i, j] = np.exp(-(jarak ** 2) / (2 * (sigma ** 2)))

# Terapkan Gaussian Lowpass Filter
fshift_filtered = fshift * mask
f_filtered = np.fft.ifftshift(fshift_filtered)
img_filtered = np.fft.ifft2(f_filtered)
img_filtered = np.abs(img_filtered)

# Buat mask untuk highpass filter Ideal
mask_ideal = np.ones((baris, kolom), np.uint8)
radius_ideal = 30
for i in range(baris):
    for j in range(kolom):
        jarak = np.sqrt((i - pusat_baris) ** 2 + (j - pusat_kolom) ** 2)
        if jarak <= radius_ideal:
            mask_ideal[i, j] = 0

# Terapkan Ideal Highpass Filter
fshift_filtered_ideal = fshift * mask_ideal
f_filtered_ideal = np.fft.ifftshift(fshift_filtered_ideal)
img_filtered_ideal = np.fft.ifft2(f_filtered_ideal)
img_filtered_ideal = np.abs(img_filtered_ideal)

# Plot gambar dan spektrum Fourier
fig, axs = plt.subplots(2, 3)
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Gambar Asli')
axs[0, 1].imshow(np.log(1 + np.abs(fshift)), cmap='gray')
axs[0, 1].set_title('Spektrum Fourier')
axs[0, 2].imshow(mask, cmap='gray')
axs[0, 2].set_title('Gaussian Lowpass Filter')
axs[1, 0].imshow(img_filtered, cmap='gray')
axs[1, 0].set_title('Gambar Setelah Filtering')
axs[1, 1].imshow(mask_ideal, cmap='gray')
axs[1, 1].set_title('Ideal Highpass Filter')
axs[1, 2].imshow(img_filtered_ideal, cmap='gray')
axs[1, 2].set_title('Gambar Setelah Filtering')
plt.show()
