import numpy as np
import cv2


def disparity(L, R, window_size=5, disparity_range=30, method=cv2.TM_SQDIFF_NORMED):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L
    Returns: Disparity map, same size as L, R
    """

    disparity_map = np.zeros(L.shape)
    height, width = L.shape
    window_row, window_col = (window_size - 1) // 2, (window_size - 1) // 2
    for r in range(height):
        tpl_row_left = max(r - window_row, 0)
        tpl_row_right = min(height, r + window_row + 1)
        for c in range(width):
            tpl_col_left = max(c - window_col, 0)
            tpl_col_right = min(width, c + window_col + 1)
            Rc_min = max(c - disparity_range // 2, 0)
            rc_max = min(c + disparity_range // 2 + 1, width)
            tpl = L[tpl_row_left:tpl_row_right, tpl_col_left:tpl_col_right].astype(np.float32)
            R_stripe = R[tpl_row_left: tpl_row_right, Rc_min:rc_max].astype(np.float32)
            error = cv2.matchTemplate(R_stripe, tpl, method)
            diff = np.arange(error.shape[0] * error.shape[1])
            diff -= max((c - Rc_min - window_col), 0)
            _, _, min_index, max_index = cv2.minMaxLoc(error)
            if method == cv2.TM_SQDIFF_NORMED:
                disparity_map[r, c] = diff[min_index[0]]
            else:
                disparity_map[r, c] = diff[max_index[0]]
    return disparity_map
