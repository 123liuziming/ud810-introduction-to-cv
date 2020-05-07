from calibrate import calibrate, projectionCenter
from fundamental import *


def Q1(inputPath):
    pts_2d = read_data(inputPath, "pts2d-norm-pic_a.txt")
    pts_3d = read_data(inputPath, "pts3d-norm.txt")
    M_least_squared = calibrate(pts_2d, pts_3d, "least_squared")
    M_SVD = calibrate(pts_2d, pts_3d, "SVD")
    C = projectionCenter(M_SVD)
    return M_least_squared, M_SVD, C


def Q2(inputPath, outputPath):
    F_lstsq, F_SVD = fundamentalMatrix(inputPath)
    F_SVD_R2 = fundamentalMatrix(inputPath, rank=2, rank3Matrix=F_lstsq)
    img_a, img_b = drawEpipolarLine(inputPath, F_SVD_R2, outputPath)
    return F_lstsq, F_SVD, F_SVD_R2, img_a, img_b


