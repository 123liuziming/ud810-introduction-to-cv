import numpy as np


def calibrate(pts_2d, pts_3d, method="least_squared"):
    # pts_2d = M.dot(pts_3d)
    # [pts_3d_x, pts_3d_y, pts_3d_z, 1, 0, 0, 0, 0, -pts_2d_x * pts_3d_x, -pts_2d_x * pts_3d_y, -pts_2d_x * pts_3d_z, -pts_2d_x]
    # [0, 0, 0, 0, pts_3d_x, pts_3d_y, pts_3d_z, 1, -pts_2d_y * pts_3d_x, -pts_2d_y * pts_3d_y, -pts_2d_y * pts_3d_z, -pts_2d_y]
    # each point makes the 2 * 12 matrix above, we call it matrix A
    # matrix A dot column vector [m_00, m_10, ..., m_23].T equals 0
    # we use least_squared function or use SVD to solve the equations

    nums_pts = pts_2d.shape[0]
    A = np.zeros((2 * nums_pts, 12), dtype=float)
    b = np.zeros(2 * nums_pts, dtype=float)
    pts_3d_x, pts_3d_y, pts_3d_z = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2]
    pts_2d_x, pts_2d_y = pts_2d[:, 0], pts_2d[:, 1]
    _zeros, _ones = np.zeros(nums_pts), np.ones(nums_pts)
    A[::2, :] = np.column_stack((pts_3d_x, pts_3d_y, pts_3d_z, _ones, _zeros, _zeros, _zeros, _zeros,
                                 -pts_2d_x * pts_3d_x, -pts_2d_x * pts_3d_y, -pts_2d_x * pts_3d_z, -pts_2d_x))
    A[1::2, :] = np.column_stack((_zeros, _zeros, _zeros, _zeros, pts_3d_x, pts_3d_y, pts_3d_z, _ones,
                                  -pts_2d_y * pts_3d_x, -pts_2d_y * pts_3d_y, -pts_2d_y * pts_3d_z, -pts_2d_y))
    M = None
    if method == "least_squared":
        A = A[:, :-1]
        b[::2] = pts_2d_x
        b[1::2] = pts_2d_y
        M, _, _, _ = np.linalg.lstsq(A, b)
        M = np.append(M, 1).reshape(3, 4)
    elif method == "SVD":
        _, _, V = np.linalg.svd(A, full_matrices=True)
        M = V.T[:, -1].reshape((3, 4))
    return M


def projectionCenter(M):
    Q, m4 = M[:, :-1], M[:, -1]
    C = -np.dot(np.linalg.inv(Q), m4)
    return C

