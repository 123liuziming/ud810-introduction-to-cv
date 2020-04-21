# ps2
import os

from disparity import *


def Q1(inputPath=".", outputPath="."):
    img1 = cv2.imread(os.path.join(inputPath, "pair0-L.png"), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(inputPath, "pair0-R.png"), cv2.IMREAD_GRAYSCALE)
    LR_map = disparity(img1, img2, window_size=11)
    RL_map = disparity(img2, img1, window_size=11)
    LR_map = cv2.normalize(LR_map, LR_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    RL_map = cv2.normalize(RL_map, RL_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(os.path.join(outputPath, "ps2-1-a-1.png"), LR_map)
    cv2.imwrite(os.path.join(outputPath, "ps2-1-a-2.png"), RL_map)
    return LR_map, RL_map


def Q2(inputPath, outputPath):
    L = cv2.imread(os.path.join(inputPath, "pair1-L.png"), cv2.IMREAD_GRAYSCALE)
    R = cv2.imread(os.path.join(inputPath, "pair1-R.png"), cv2.IMREAD_GRAYSCALE)
    D_L = np.abs(disparity(L, R, 7, 100))
    D_R = np.abs(disparity(R, L, 7, 100))
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(os.path.join(outputPath, "ps2-2-a-1.png"), D_L)
    cv2.imwrite(os.path.join(outputPath, "ps2-2-a-2.png"), D_R)

    return D_L, D_R


def Q3(inputPath, outputPath):
    L = cv2.imread(os.path.join(inputPath, "pair2-L.png"), cv2.IMREAD_GRAYSCALE)
    R = cv2.imread(os.path.join(inputPath, "pair2-R.png"), cv2.IMREAD_GRAYSCALE)
    L = cv2.pyrDown(cv2.GaussianBlur(cv2.equalizeHist(L), (5,) * 2, 3))
    R = cv2.pyrDown(cv2.GaussianBlur(cv2.equalizeHist(R), (5,) * 2, 3))
    L = L * 1.0 / 255.0
    R = R * 1.0 / 255.0
    D_L = np.abs(disparity(L, R, 7, 100, cv2.TM_CCORR_NORMED))
    D_R = np.abs(disparity(R, L, 7, 100, cv2.TM_CCORR_NORMED))
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_L = cv2.pyrUp(D_L)
    D_R = cv2.pyrUp(D_R)
    cv2.imwrite(os.path.join(outputPath, "ps2-3-a-1.png"), D_L)
    cv2.imwrite(os.path.join(outputPath, "ps2-3-a-2.png"), D_R)
    return D_L, D_R