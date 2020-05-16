import cv2
import numpy as np


def harris_matrix(Ix, Iy, window_size=5, alpha=0.04, norm=False):
    Ixx, Ixy, Iyy = Ix * Ix, Ix * Iy, Iy * Iy
    # get the weights function w
    w = np.zeros((window_size, window_size), dtype=np.float32)
    w[window_size // 2, window_size // 2] = 1.0
    w = cv2.GaussianBlur(w, (window_size, window_size), 0)
    height, width = Ix.shape
    # print(height, width)
    res = np.zeros_like(Ix)
    for r in range(window_size // 2, height - window_size // 2):
        min_r = max(0, r - window_size // 2)
        max_r = min(height, min_r + window_size)
        for c in range(window_size // 2, width - window_size // 2):
            min_c = max(0, c - window_size // 2)
            max_c = min(width, min_c + window_size)
            # print(min_c, max_c)
            M = np.array(
                (np.sum(Ixx[min_r:max_r, min_c:max_c] * w),
                 np.sum(Ixy[min_r:max_r, min_c:max_c] * w),
                 np.sum(Ixy[min_r:max_r, min_c:max_c] * w),
                 np.sum(Iyy[min_r:max_r, min_c:max_c] * w),
                 )
            ).reshape(2, 2)
            res[r, c] = np.linalg.det(M) - alpha * (np.trace(M) ** 2)
    if norm:
        res = cv2.normalize(res, res, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return res


def harris_corners(img, res, threshold=1e-3, neighbour_size=5):
    img = np.float32(img)
    height, width = res.shape
    res = res * (res > (threshold * np.max(res))) * (res > 0)
    # print(res.shape)
    rows, cols = np.nonzero(res)
    for r, c in zip(rows, cols):
        minr = max(0, r - neighbour_size // 2)
        maxr = min(height, minr + neighbour_size)
        minc = max(0, c - neighbour_size // 2)
        maxc = min(width, minc + neighbour_size)
        if res[r, c] == np.max(res[minr:maxr, minc:maxc]) and res[r, c] > 0:
            img = cv2.circle(img, (r, c), 5, (0, 0, 255))
    return img
