import matplotlib.pyplot as plt
from skimage import data, transform
import numpy as np

## Task 1
cameraman = data.camera()
inverted_cameraman = 255 - cameraman
rotated_cameraman = transform.rotate(cameraman, angle=180)
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(cameraman, cmap='gray')
ax[0].set_title('Original Cameraman')
ax[0].axis('off')
ax[1].imshow(inverted_cameraman, cmap='gray')
ax[1].set_title('Inverted Cameraman')
ax[1].axis('off')
ax[2].imshow(rotated_cameraman, cmap='gray')
ax[2].set_title('Rotated Cameraman')
ax[2].axis('off')
plt.show()  


## Task 2
rgb_astronut = data.astronaut()
red_astronut = rgb_astronut[:, :, 0]
green_astronut = rgb_astronut[:, :, 1]
blue_astronut = rgb_astronut[:, :, 2]
titles = ['Original', 'Red', 'Green', 'Blue']
cmaps = ['gray', 'Reds', 'Greens', 'Blues']
fig, axes = plt.subplots(2, 2, figsize=(15, 5))

axes[0, 0].imshow(rgb_astronut)
axes[0, 0].set_title(titles[0])
axes[0, 0].axis('off')

axes[0, 1].imshow(red_astronut, cmap=cmaps[1])
axes[0, 1].set_title(titles[1])
axes[0, 1].axis('off')

axes[1, 0].imshow(green_astronut, cmap=cmaps[2])
axes[1, 0].set_title(titles[2])
axes[1, 0].axis('off')  

axes[1, 1].imshow(blue_astronut, cmap=cmaps[3])
axes[1, 1].set_title(titles[3])
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()


#Task 3
def my_hist(I, nbins):
    hist, _ =  np.histogram(I, bins=nbins, range=(0, 256))
    return hist

def hist_linear(I, range_min, range_max):
    min_pixel = np.min(I)
    max_pixel = np.max(I)

    if max_pixel == min_pixel:
        raise ValueError("All pixel values are the same. Linear stretching is not possible.")

    normalized = (I - min_pixel) / (max_pixel - min_pixel)
    g_x = normalized * (range_max - range_min) + range_min
    return g_x.astype(np.uint8)

def hist_gamma(I, gamma):
    if gamma <= 0:
        raise ValueError("Gamma must be greater than 0.")
    else:
        g_x = 255 * (I / 255) ** gamma

    return np.clip(g_x, 0, 255).astype(np.uint8)

def hist_equalization(I):
    I_int = I.astype(np.uint8)
    hist, bins = np.histogram(I.flatten(), bins=256, range=[0, 256])
    h_k = hist / I_int.size
    cdf =  h_k.cumsum()
    lookup_table = np.floor(255 * cdf).astype(np.uint8)
    return lookup_table[I_int]

origin_img = rgb_astronut
linear_img = hist_linear(origin_img, 0, 255)
gamma_img = hist_gamma(origin_img, 2.2)
equalized_img = hist_equalization(origin_img)

data_list = [
        (origin_img, "Original"),
        (linear_img, "Linear Stretched"),
        (gamma_img, "Gamma Corrected (2.0)"),
        (equalized_img, "Equalized")
    ]

fig, axes = plt.subplots(4, 2, figsize=(12, 8))

for i, (img, title) in enumerate(data_list):
    axes[i, 0].imshow(img, cmap='gray')
    axes[i, 0].set_title(title)
    axes[i, 0].axis('off')

    hist = my_hist(img, nbins=256)
    axes[i, 1].bar(range(256), hist, color='gray')
    axes[i, 1].set_title(f"{title} Histogram")
    axes[i, 1].set_xlim(0, 255) 
    
plt.tight_layout()
plt.show()