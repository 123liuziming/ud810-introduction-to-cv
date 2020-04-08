"""
Input images
Find two interesting images to use. They should be color, rectangular in shape (NOT square). Pick one that is wide and one tall.
You might find some classic vision examples here. Or take your own. Make sure the image width or height do not exceed 512 pixels.
Output: Store the two images as ps0-1-a-1.png and ps0-1-a-2.png inside the output folder
"""

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


def Q1(outputPath, img1, img2):
    
    cv2.imwrite(os.path.join(outputPath, "ps0-1-a-1.png"), img1)
    cv2.imwrite(os.path.join(outputPath, "ps0-1-a-2.png"), img2)



"""
Color  planes
Swap the red and blue pixels of image 1
Output: Store as ps0-2-a-1.png in the output folder
Create a monochrome image (img1_green) by selecting the green channel of image 1
Output: ps0-2-b-1.png
Create a monochrome image (img1_red) by selecting the red channel of image 1
Output: ps0-2-c-1.png
Which looks more like what you’d expect a monochrome image to look like? Would you expect a computer vision algorithm to work on one better than the other?
Output: Text response in report ps0_report.pdf
"""

def Q2(outputPath, img1):

    # B G R -> R G B
    _, img1Green, img1Red = cv2.split(img1)
    img1 = img1[:, :, (2, 1, 0)]
    cv2.imwrite(os.path.join(outputPath, "ps0-2-a-1.png"), img1)
    cv2.imwrite(os.path.join(outputPath, "ps0-2-b-1.png"), img1Green)
    cv2.imwrite(os.path.join(outputPath, "ps0-2-c-1.png"), img1Red)

    return img1, img1Green, img1Red


"""
Replacement of pixels (Note: For this, use the better channel from 2-b/2-c as monochrome versions.)
Take the inner center square region of 100x100 pixels of monochrome version of image 1 and insert them into the center of monochrome version of image 2
Output: Store the new image created as ps0-3-a-1.png
"""
def Q3(outputPath, img1, img2):
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    mid = img1[h1 // 2 - 50 : h1 // 2 + 50,  w1 // 2 - 50 : w1 // 2 + 50]
    img2[h2 // 2 - 50 : h2 // 2 + 50,  w2 // 2 - 50 : w2 // 2 + 50] = mid
    cv2.imwrite(os.path.join(outputPath, "ps0-3-a-1.png"), img2)
    
    return img2

"""
Arithmetic and Geometric operations
What is the min and max of the pixel values of img1_green? What is the mean? What is the standard deviation?  And how did you compute these?
Output: Text response, with code snippets
Subtract the mean from all pixels, then divide by standard deviation, then multiply by 10 (if your image is 0 to 255) or by 0.05 (if your image ranges from 0.0 to 1.0). Now add the mean back in.
Output: ps0-4-b-1.png
Shift img1_green to the left by 2 pixels.
Output: ps0-4-c-1.png
Subtract the shifted version of img1_green from the original, and save the difference image.
Output: ps0-4-d-1.png (make sure that the values are legal when you write the image so that you can see all relative differences), text response: What do negative pixel values mean anyways?
"""

def Q4(outputPath, img1Green):
    minPixel = np.min(img1Green)
    maxPixel = np.max(img1Green)
    meanPixel = np.mean(img1Green)
    stDeviation = np.std(img1Green)
    w, h = img1Green.shape

    img1Greenb = cv2.add(cv2.multiply(cv2.divide(cv2.subtract(
        img1Green, meanPixel), stDeviation), 10), meanPixel)
    cv2.imwrite(os.path.join(outputPath, "ps0-4-b-1.png"), img1Greenb)

    img1Greenc = img1Greenb.copy()
    mat_left2 = np.float32([[1, 0, -2], [0, 1, 0]])
    img1Greenc = cv2.warpAffine(img1Greenc, mat_left2, (h, w))
    cv2.imwrite(os.path.join(outputPath, "ps0-4-c-1.png"), img1Greenc)

    img1Greend = cv2.subtract(img1Green, img1Greenc)
    cv2.imwrite(os.path.join(outputPath, "ps0-4-d-1.png"), img1Greend)

    img1Greenb = cv2.cvtColor(img1Greenb, cv2.COLOR_GRAY2BGR)
    img1Greenc = cv2.cvtColor(img1Greenc, cv2.COLOR_GRAY2BGR)
    img1Greend = cv2.cvtColor(img1Greend, cv2.COLOR_GRAY2BGR)

    return minPixel, maxPixel, meanPixel, stDeviation, img1Greenb, img1Greenc, img1Greend


"""
Noise
Take the original colored image (image 1) and start adding Gaussian noise to the pixels in the green channel. Increase sigma until the noise is somewhat visible.  
Output: ps0-5-a-1.png, text response: What is the value of sigma you had to use?
Now, instead add that amount of noise to the blue channel.
Output: ps0-5-b-1.png
Which looks better? Why?
Output: Text response
"""
def Q5(outputPath, img1):
    # B G R
    imgBlur1 = img1.copy()
    imgBlur1[:, :, 1] = cv2.GaussianBlur(img1[:, :, 1], (11, 11), 5)
    imgBlur2 = img1.copy()
    imgBlur2[:, :, 0] = cv2.GaussianBlur(img1[:, :, 0], (11, 11), 5)
    cv2.imwrite(os.path.join(outputPath, "ps0-5-a-1.png"), imgBlur1)
    cv2.imwrite(os.path.join(outputPath, "ps0-5-b-1.png"), imgBlur2)
    
    return imgBlur1, imgBlur2


