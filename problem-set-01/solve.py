import os

from hough_lines_acc import *
from hough_lines_draw import *
from hough_circles_draw import *
from find_circles import *


def canny_sigma(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def Q1(inputPath, outputPath):
    img1Path = os.path.join(inputPath, "ps1-input0.png")
    img = cv2.imread(img1Path, cv2.IMREAD_GRAYSCALE)
    canny = cv2.Canny(img, 100, 200)
    cv2.imwrite(os.path.join(outputPath, "ps1-1-a-1.png"), canny)
    return canny


def Q2(inputPath, outputPath, canny_img):
    originImg = cv2.imread(os.path.join(inputPath, "ps1-input0.png"))

    H, thetas, rhos = hough_line(canny_img)
    cv2.imwrite(os.path.join(outputPath, "ps1-2-a-1.png"), H)

    peaks = hough_peaks(H, 10)
    img_line = draw_lines(originImg, peaks, os.path.join(outputPath, "ps1-3-a-1.png"))

    return H, peaks, img_line


def Q3(inputPath, outputPath, sigma=5):
    img_noise = cv2.imread(os.path.join(inputPath, "ps1-input0-noise.png"))

    # smooth image
    img_smoothed = cv2.GaussianBlur(img_noise, (23, 23), sigma)
    cv2.imwrite(os.path.join(outputPath, "ps1-3-a-1.png"), img_smoothed)

    # edge image
    canny_origin = cv2.Canny(img_noise, 20, 40)
    canny_smoothed = cv2.Canny(img_smoothed, 20, 40)
    cv2.imwrite(os.path.join(outputPath, "ps1-3-b-1.png"), canny_origin)
    cv2.imwrite(os.path.join(outputPath, "ps1-3-b-2.png"), canny_smoothed)

    H, _, rhos = hough_line(canny_smoothed)
    peaks = hough_peaks(H, num_peaks=10, threshold=200, nhood_size=30)
    img_line = draw_lines(img_noise, peaks, rhos, _, os.path.join(outputPath, "ps1-3-c-2.png"))

    return img_smoothed, canny_origin, canny_smoothed, H, peaks, img_line


def Q4(inputPath, outputPath):
    origin_img = cv2.imread(os.path.join(inputPath, "ps1-input1.png"))
    img_smooth = cv2.GaussianBlur(cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY), (25, 25), 3)
    cv2.imwrite(os.path.join(outputPath, "ps1-4-a-1.png"), img_smooth)
    img_canny = canny_sigma(img_smooth, 0.8)
    cv2.imwrite(os.path.join(outputPath, "ps1-4-b-1.png"), img_canny)
    H, _, rhos = hough_line(img_canny)
    peaks = hough_peaks(H, num_peaks=20, threshold=150, nhood_size=35)
    img_line = draw_lines(origin_img.copy(), peaks, rhos, _, os.path.join(outputPath, "ps1-4-c-2.png"))

    return origin_img, img_smooth, img_canny, H, img_line, peaks


def Q5(inputPath, outputPath):
    origin_img = cv2.imread(os.path.join(inputPath, "ps1-input1.png"))
    img_smooth = cv2.GaussianBlur(cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY), (25, 25), 3)
    cv2.imwrite(os.path.join(outputPath, "ps1-5-a-1.png"), img_smooth)
    img_canny = canny_sigma(img_smooth, 0.8)
    cv2.imwrite(os.path.join(outputPath, "ps1-5-a-2.png"), img_canny)
    H_20 = hough_circles_acc(img_canny, 20)
    peaks_20 = hough_peaks(H_20, 10, 140, 100)
    pic_circle_20 = hough_circles_draw(origin_img.copy(), os.path.join(outputPath, "ps1-5-a-3.png"), peaks_20, 20)
    centers, radius = find_circles(img_canny, [20, 50], threshold=130, nhood_size=10)
    img_circles = origin_img.copy()
    for i in range(len(radius)):
        img_circles = hough_circles_draw(img_circles, os.path.join(outputPath, "ps1-5-b-1.png"),
                                         centers[i], radius[i])
    return origin_img, img_smooth, img_canny, pic_circle_20, img_circles
