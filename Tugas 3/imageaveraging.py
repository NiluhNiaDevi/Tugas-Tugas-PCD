import cv2
import numpy as np

# Baca dua gambar
img1 = cv2.imread('iloh1.jpeg')
img2 = cv2.imread('iloh2.jpeg')

# Ubah ukuran gambar agar sama
img1_resized = cv2.resize(img1, (640, 480))
img2_resized = cv2.resize(img2, (640, 480))

# Gabungkan kedua gambar menjadi satu array
img_array = np.array([img1_resized, img2_resized])

# Hitung nilai rata-rata dari setiap pixel
average_image = np.average(img_array, axis=0)

# Konversi tipe data hasil perhitungan ke tipe data unsigned integer 8 bit
average_image = average_image.astype(np.uint8)

# Tampilkan gambar hasil averaging
cv2.imshow('Average Image', average_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
