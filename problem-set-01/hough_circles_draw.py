import cv2


def hough_circles_draw(img, outputPath, peaks, radius):
    for peak in peaks:
        cv2.circle(img, (peak[0], peak[1]), radius, (0, 255, 0), 2)
    cv2.imwrite(outputPath, img)
    return img
