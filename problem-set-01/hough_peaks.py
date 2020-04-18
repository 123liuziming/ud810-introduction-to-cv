import numpy as np
import cv2


def clip(idx):
    return int(max(idx, 0))


def hough_peaks(H, num_peaks=1, threshold=100, nhood_size=5):
    peaks = np.zeros((num_peaks, 2), dtype=np.uint64)
    H_cpy = H.copy()
    height, width = H.shape
    for i in range(num_peaks):
        _, max_val, _, max_loc = cv2.minMaxLoc(H_cpy)
        if max_val > threshold:
            peaks[i] = max_loc
            (c, r) = max_loc
            t = nhood_size // 2.0
            H_cpy[clip(r - t):int(r + t + 1), clip(c - t):int(c + t + 1)] = 0
            if c + t + 1 >= width:
                H_cpy[clip(r - t):int(r + t + 1), 0: int(c + t - width)] = 0
            if c - t < 0:
                H_cpy[clip(r - t):int(r + t + 1), int(width + c - t): int(width)] = 0
        else:
            peaks = peaks[:i]
            break
    return peaks[:, ::-1]
