import os
from SIFT import *
from harris_corner import *

# Harris corners
# To find the Harris points you need to compute the gradients in both the X and Y directions.
# These will probably have to be lightly filtered using a Gaussian to be well behaved.
# You can do this either the “naive” way
# filter the image and then do simple difference between left and right (X gradient) or up and down (Y gradient)
# or you can take an analytic derivative of a Gaussian in X or Y and use that filter.
# The scale of the filtering is up to you.
# You may play with the size of the Gaussian as it will interact with the window size of the corner detection.

img_list = ["transA.jpg", "transB.jpg", "simA.jpg", "simB.jpg"]


def Q1(inputPath, outputPath):
    """
    Write functions to compute both the X and Y gradients. Try your code on both transA and simA. To display the
    output, adjoin the two gradient images(X and Y) to make a new, twice as wide, single image (the "gradient-pair").
    Since gradients have negative and positive values, you’ll need to produce an image that is gray for 0.0 and black
    is negative and white is positive.
    """

    transA = cv2.imread(os.path.join(inputPath, "transA.jpg"),cv2.IMREAD_GRAYSCALE)
    simA = cv2.imread(os.path.join(inputPath, "simA.jpg"), cv2.IMREAD_GRAYSCALE)
    transA_grad_x = calc_grad(transA, direction="x", norm=True)
    transA_grad_y = calc_grad(transA, direction="y", norm=True)
    simA_grad_x = calc_grad(simA, direction="x", norm=True)
    simA_grad_y = calc_grad(simA, direction="y", norm=True)
    trans_stack = np.hstack((transA_grad_x, transA_grad_y))
    sim_stack = np.hstack((simA_grad_x, simA_grad_y))
    cv2.imwrite(os.path.join(outputPath, "ps4-1-a-1.png"), trans_stack)
    cv2.imwrite(os.path.join(outputPath, "ps4-1-a-2.png"), sim_stack)

    # compute the Harris value

    for idx, name in enumerate(img_list):
        I_x = calc_grad(cv2.imread(os.path.join(inputPath, name), cv2.IMREAD_GRAYSCALE), direction="x")
        I_y = calc_grad(cv2.imread(os.path.join(inputPath, name), cv2.IMREAD_GRAYSCALE), direction="y")
        H = harris_matrix(I_x, I_y, norm=True)
        cv2.imwrite(os.path.join(outputPath, "ps4-1-b-" + str(idx + 1) + ".png"), H)

    for idx, name in enumerate(img_list):
        I_x = calc_grad(cv2.imread(os.path.join(inputPath, name), cv2.IMREAD_GRAYSCALE), direction="x")
        I_y = calc_grad(cv2.imread(os.path.join(inputPath, name), cv2.IMREAD_GRAYSCALE), direction="y")
        H = harris_matrix(I_x, I_y, norm=False)
        corner_img = harris_corners(cv2.imread(os.path.join(inputPath, name), cv2.IMREAD_GRAYSCALE), H)
        cv2.imwrite(os.path.join(outputPath, "ps4-2-c-" + str(idx + 1) + ".png"), corner_img)


def Q2(inputPath, outputPath):
    for idx, name in enumerate(img_list):
        img = draw_interest_point(cv2.imread(os.path.join(inputPath, name), cv2.IMREAD_GRAYSCALE))
        cv2.imwrite(os.path.join(outputPath, "ps4-2-a-" + str(idx + 1) + ".png"), img)
    for idx in range(len(img_list) // 2):
        img1 = cv2.imread(os.path.join(inputPath, img_list[2 * idx]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(inputPath, img_list[2 * idx + 1]), cv2.IMREAD_GRAYSCALE)
        matched_image = match_interest_point(img1, img2)
        cv2.imwrite(os.path.join(outputPath, "ps4-2-b-" + str(idx + 1) + ".png"), matched_image)
