from read_data import read_data
import numpy as np
import cv2
import os


def solveEquations(pts_a, pts_b, method = "least_square"):
    # we have [pts_b_x, pts_b_y, 1] dot Matrix F dot [pts_a_x, pts_a_y, 1].T equals 0
    # =>[pts_b_x * pts_a_x, pts_b_x * pts_a_y, pts_b_x, pts_b_y * pts_a_x, pts_b_y * pts_a_y, pts_b_y, pts_a_x, pts_a_y, 1]
    # dot [F_11 ... F_33] equals 0
    # we use least_square or SVD to solve this equations

    num_pts = pts_a.shape[0]
    pts_a_x, pts_a_y = pts_a[:, 0], pts_a[:, 1]
    pts_b_x, pts_b_y = pts_b[:, 0], pts_b[:, 1]
    A = np.column_stack((pts_b_x * pts_a_x, pts_b_x * pts_a_y, pts_b_x, pts_b_y * pts_a_x, pts_b_y * pts_a_y, pts_b_y, pts_a_x, pts_a_y, np.ones(num_pts)))
    F = None
    if method == "least_square":
        A = A[:, :-1]
        b = -np.ones(num_pts)
        F, _, _, _ = np.linalg.lstsq(A, b, rcond=-1)
        F = np.append(F, 1).reshape((3, 3))
    elif method == "SVD":
        _, _, V = np.linalg.svd(A)
        F = V.T[-1]
    return F


def fundamentalMatrix(inputPath, rank=3, rank3Matrix=None):
    if rank == 3:
        pts_a, pts_b = read_data(inputPath, "pts2d-pic_a.txt"), read_data(inputPath, "pts2d-pic_b.txt")
        F_lstsq = solveEquations(pts_a, pts_b, method="least_square")
        F_SVD = solveEquations(pts_a, pts_b, method="SVD")
        return F_lstsq, F_SVD
    elif rank == 2:
        U, S, V = np.linalg.svd(rank3Matrix)
        S[-1] = 0
        S = np.diag(S)
        F = np.dot(np.dot(U, S), V)
        return F
    return None


def drawEpipolarLine(inputPath, F, outputPath):
    img_a = cv2.imread(os.path.join(inputPath, "pic_a.jpg"))
    img_b = cv2.imread(os.path.join(inputPath, "pic_b.jpg"))
    pts_a = read_data(inputPath, "pts2d-pic_a.txt")
    pts_b = read_data(inputPath, "pts2d-pic_b.txt")
    pts_a = np.column_stack((pts_a, np.ones(pts_a.shape[0])))
    pts_b = np.column_stack((pts_b, np.ones(pts_a.shape[0])))
    eplines_a = np.dot(pts_b, F)
    eplines_b = np.dot(pts_a, F.T)
    height, width, _ = img_a.shape
    boundary_l = np.cross([0, 0, 1], [height - 1, 0, 1])
    boundary_r = np.cross([0, width - 1, 1], [height - 1, width - 1, 1])
    for line_a, line_b in zip(eplines_a, eplines_b):
        pts1 = np.cross(line_a, boundary_l)
        pts2 = np.cross(line_a, boundary_r)
        pts1 /= pts1[2]
        pts2 /= pts2[2]
        cv2.line(img_a, tuple(pts1[:2].astype(int)), tuple(pts2[:2].astype(int)), (0, 255, 0), thickness=2)
        pts1 = np.cross(line_b, boundary_l)
        pts2 = np.cross(line_b, boundary_r)
        pts1 /= pts1[2]
        pts2 /= pts2[2]
        cv2.line(img_b, tuple(pts1[:2].astype(int)), tuple(pts2[:2].astype(int)), (0, 255, 0), thickness=2)
    cv2.imwrite(os.path.join(outputPath, "ps3-2-c-1.png"), img_a)
    cv2.imwrite(os.path.join(outputPath, "ps3-2-c-2.png"), img_b)
    return img_a, img_b
