import cv2
import numpy as np


def calc_grad(img, direction="x", ksize=3, norm=False):
    grad = None
    if direction == "x":
        grad = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize)
    elif direction == "y":
        grad = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize)
    if norm:
        grad = cv2.normalize(grad, grad, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return grad


def calc_grad_orientation(Ix, Iy):
    return np.arctan2(Iy, Ix)


def calc_grad_magnitude(Ix, Iy):
    return np.sqrt(Ix ** 2 + Iy ** 2)

