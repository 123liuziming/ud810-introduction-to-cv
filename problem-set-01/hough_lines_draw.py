import numpy as np
import cv2


def draw_lines(img, peaks, _rho, theta, outputPath):
    max_rho = int(_rho[-1])
    if len(img.shape) == 3:
        height, width, _ = img.shape
    else:
        height, width = img.shape
    for rho, theta in peaks:
        # print("rho is %d" % rho)
        # print("theta is %d" % theta)
        rho = int(rho - max_rho)
        if theta == 0:
            y0, yn = 0, height - 1
            x0, xn = rho, rho
        elif theta == 90:
            x0, xn = 0, width - 1
            y0, yn = rho, rho
        else:
            if rho > 0:
                x0, y0 = 0, int(rho / np.sin(np.deg2rad(theta)))
                xn, yn = int(rho / np.cos(np.deg2rad(theta))), 0
            else:
                xn = width - 1
                yn = int((rho - xn * np.cos(np.deg2rad(theta))) / np.sin(np.deg2rad(theta)))
                x0, y0 = int(rho / np.cos(np.deg2rad(theta))), 0
        cv2.line(img, (x0, y0), (xn, yn), (0, 255, 0), 2)
    cv2.imwrite(outputPath, img)
    return img
