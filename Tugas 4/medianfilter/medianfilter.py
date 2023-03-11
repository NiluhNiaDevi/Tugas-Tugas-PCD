import cv2

# membaca gambar
img = cv2.imread('buah.jpeg', cv2.IMREAD_GRAYSCALE)

# menerapkan Median Filter dengan kernel size 3x3
median = cv2.medianBlur(img, 3)

# menampilkan gambar asli dan hasil Median Filter
cv2.imshow('Original Image', img)
cv2.imshow('Median Filter', median)
cv2.waitKey(0)
cv2.destroyAllWindows()
