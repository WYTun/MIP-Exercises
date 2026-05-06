import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from skimage import data, transform



def resize_copy(I, factor):
    rescaled_rows = np.repeat(0, I.shape[0], factor)
    I_resized = np.repeat(I, rescaled_rows, axis=1)

    return I_resized

def zero_padding(I, factor):
    h, w = I.shape
    new_h, new_w = h * factor, w * factor
    I_fft = fftshift(fft2(I))

    padded_fft = np.zeros((new_h, new_w), dtype=complex)

    start_h = (new_h - h) // 2
    start_w = (new_w - w) // 2
    padded_fft[start_h:start_h + h, start_w:start_w + w] = I_fft

    I_zeropad = np.real(ifft2(ifftshift(padded_fft))) * (factor ** 2)
    return I_zeropad

