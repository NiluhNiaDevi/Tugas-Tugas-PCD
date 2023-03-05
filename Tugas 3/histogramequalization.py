import cv2
import matplotlib.pyplot as plt

# Load the images
img_low = cv2.imread('foto_rendah.jpg', 0)
img_high = cv2.imread('foto_tinggi.jpg', 0)
img_normal = cv2.imread('foto_sedang.jpg', 0)

# Perform histogram equalization
img_eq_low = cv2.equalizeHist(img_low)
img_eq_high = cv2.equalizeHist(img_high)
img_eq_normal = cv2.equalizeHist(img_normal)

# Plot the original and equalized images
fig, axs = plt.subplots(3, 3, figsize=(15,10))
axs[0, 0].imshow(img_low, cmap='gray')
axs[0, 0].set_title('Foto Rendah')
axs[0, 1].hist(img_low.ravel(),256,[0,256], color='red')
axs[0, 2].imshow(img_eq_low, cmap='gray')
axs[0, 2].set_title('Foto Rendah Equalized')
axs[1, 0].imshow(img_high, cmap='gray')
axs[1, 0].set_title('Foto Tinggi')
axs[1, 1].hist(img_high.ravel(),256,[0,256], color='red')
axs[1, 2].imshow(img_eq_high, cmap='gray')
axs[1, 2].set_title('Foto Tinggi Equalized')
axs[2, 0].imshow(img_normal, cmap='gray')
axs[2, 0].set_title('Foto Normal')
axs[2, 1].hist(img_normal.ravel(),256,[0,256], color='red')
axs[2, 2].imshow(img_eq_normal, cmap='gray')
axs[2, 2].set_title('Foto Normal Equalized')
plt.show()
