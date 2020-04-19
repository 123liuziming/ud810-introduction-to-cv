import numpy as np


def hough_circles_acc(img_edges, radius):
    m, n = img_edges.shape
    accumulator = np.zeros((m, n))
    theta = np.arange(0, 360)
    y_idx, x_idx = np.nonzero(img_edges)
    for i in range(len(x_idx)):
        x, y = x_idx[i], y_idx[i]
        b = (y - radius * np.sin(np.deg2rad(theta))).astype(np.uint)
        a = (x - radius * np.cos(np.deg2rad(theta))).astype(np.uint)
        mask = (a < m) & (b < n)
        a, b = a[mask], b[mask]
        c = np.stack([a, b], 1)
        cc = np.ascontiguousarray(c).view(np.dtype((np.void, c.dtype.itemsize * c.shape[1])))
        _, idxs, counts = np.unique(cc, return_index=True, return_counts=True)
        uc = c[idxs].astype(np.uint)
        accumulator[uc[:, 0], uc[:, 1]] += counts
    return accumulator
