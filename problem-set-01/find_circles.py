from hough_circles_acc import *
from hough_peaks import *


def find_circles(edge_img, radius_range=None, threshold=140, nhood_size=10):
    if radius_range is None:
        radius_range = [1, 2]
    n = radius_range[1] - radius_range[0]
    H_size = (n,) + edge_img.shape
    H = np.zeros(H_size, dtype=np.uint)
    centers = ()
    radius = np.arange(radius_range[0], radius_range[1])
    valid_radius = np.array([], dtype=np.uint)
    for i in range(len(radius)):
        H = hough_circles_acc(edge_img, radius[i])
        peaks = hough_peaks(H, num_peaks=10, threshold=threshold,
                            nhood_size=nhood_size)
        if peaks.size:
            valid_radius = np.append(valid_radius, radius[i])
            centers = centers + (peaks,)
    centers = np.array(centers)
    return centers, valid_radius.astype(np.uint)
