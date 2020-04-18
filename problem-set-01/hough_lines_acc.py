import numpy as np
import cv2

"""
Write a function hough_lines_acc that computes the Hough Transform for lines and produces an accumulator array. Your code should conform to the specifications of the Matlab function hough: http://www.mathworks.com/help/images/ref/hough.html
Note that it has two optional parameters RhoResolution and Theta, and returns three values - the hough accumulator array H, theta (θ) values that correspond to columns of H and rho (ρ) values that correspond to rows of H.
"""


def hough_line(img, rho_scale=1, thetas=np.arange(-90, 90, 1)):

    rho_max = int(np.linalg.norm(img.shape, ord=2)) + 1
    rho = np.arange(0, rho_max)
    num_rho = int(rho_max / rho_scale)

    # set thetas to [0, max - min)
    thetas -= np.min(thetas)

    # init accumulator array
    accumulator = np.zeros((2 * num_rho + 1, np.max(thetas) + 1))

    y_index, x_index = np.nonzero(img)
    for idx in range(len(x_index)):
        x, y = x_index[idx], y_index[idx]
        d = (x * np.cos(np.deg2rad(thetas)) + y * np.sin(np.deg2rad(thetas))).astype(int)
        d += rho_max
        d = (d / rho_scale).astype(int)
        accumulator[d, thetas] += 1
    accumulator = cv2.normalize(accumulator, accumulator, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    return accumulator, thetas, rho
