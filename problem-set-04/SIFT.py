from gradient_util import *


def draw_interest_point(pic, line_long=25):
    """
    Write the function to compute the angle. Then for the set of interest points you found above, plot the points for
    all of transA, transB, simA and simB on the respective images and draw a little line that shows the direction of the
    gradient.
    """
    I_x = calc_grad(pic, direction="x")
    I_y = calc_grad(pic, direction="y")
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(pic, None)
    pic = cv2.drawKeypoints(pic, kp, pic)
    grad = calc_grad_orientation(I_x, I_y)
    for p in kp:
        x_2 = int(p.pt[0]) + line_long
        pt = tuple(map(lambda x: int(x), p.pt))
        cv2.line(pic, pt, (x_2, grad[pt] * (x_2 - pt[0]) + pt[1]), (0, 0, 255))
    return pic


def match_interest_point(pic1, pic2):
    sift = cv2.xfeatures2d.SIFT_create()
    key_pts1, desc1 = sift.detectAndCompute(pic1, None)
    key_pts2, desc2 = sift.detectAndCompute(pic1, None)
    bfm = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = sorted(bfm.knnMatch(desc1, desc2, k=1))
    matched_image = None
    matched_image = cv2.drawMatches(pic1, key_pts1, pic2, key_pts2, matches[:10],
                                    flags=2, outImg=matched_image)
    return matched_image
