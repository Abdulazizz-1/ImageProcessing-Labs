import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'jellyfish.png'
original_img = cv2.imread(image_path, 0)

if original_img is None:
    print(f"Error: Couldn't read the image at {image_path}. Please check the filename and path.")
    exit()

r_min = np.min(original_img)
r_max = np.max(original_img)

print(f"r_min: {r_min}")
print(f"r_max: {r_max}")

lut = np.arange(256, dtype='uint8')

denominator = float(r_max - r_min)
if denominator == 0:
    denominator = 1
stretching_factor = 255.0 / denominator

lut_stretching = np.clip((lut - r_min) * stretching_factor, 0, 255).astype('uint8')

stretched_img = cv2.LUT(original_img, lut_stretching)

hist_original = cv2.calcHist([original_img], [0], None, [256], [0, 256])
hist_stretched = cv2.calcHist([stretched_img], [0], None, [256], [0, 256])

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(original_img, cmap='gray')
plt.title(f"Original Image (r_min={r_min}, r_max={r_max})")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.plot(hist_original, color='blue')
plt.title("Original Histogram")
plt.xlim([0, 256])
plt.grid(True)

plt.subplot(2, 2, 3)
plt.imshow(stretched_img, cmap='gray')
plt.title("After Contrast Stretching (0-255)")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.plot(hist_stretched, color='red')
plt.title("Stretched Histogram")
plt.xlim([0, 256])
plt.grid(True)

plt.tight_layout()
plt.show()