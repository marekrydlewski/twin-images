from skimage.color import rgb2gray
import numpy as np
import cv2


def test(extract, compare, keypoints1, keypoints2):
    y_true = []
    y_scores = []
    for i in range(1, 351):
        img1 = rgb2gray(cv2.imread('positive-' + i + '-1', 0))
        img2 = rgb2gray(cv2.imread('positive-' + i + '-2', 0))

        desc1 = extract(img1, keypoints1)
        desc2 = extract(img2, keypoints2)

        y_true.append(1)
        y_scores.append(compare(desc1, desc2))

    for i in range(1, 351):
        img1 = rgb2gray(cv2.imread('negative-' + i + '-1', 0))
        img2 = rgb2gray(cv2.imread('negative-' + i + '-2', 0))

        desc1 = extract(img1, keypoints1)
        desc2 = extract(img2, keypoints2)

        y_true.append(0)
        y_scores.append(compare(desc1, desc2))

    return y_true, y_scores

