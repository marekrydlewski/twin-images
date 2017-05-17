from skimage.color import rgb2gray
from sklearn.metrics import roc_auc_score
import numpy as np
import cv2
import brief
import gradient
import bright

def test(extract, compare, keypoints1, keypoints2):
    y_true = []
    y_scores = []
    for i in range(1, 350):
        img1 = rgb2gray(cv2.imread('test/positive-p' + str(i) + '-1.png', 0))
        img2 = rgb2gray(cv2.imread('test/positive-p' + str(i) + '-2.png', 0))

        desc1 = extract(img1, keypoints1)
        desc2 = extract(img2, keypoints2)

        y_true.append(0)
        y_scores.append(compare(desc1[1], desc2[1]))

    for i in range(1, 350):
        img1 = rgb2gray(cv2.imread('test/negative-p' + str(i) + '-1.png', 0))
        img2 = rgb2gray(cv2.imread('test/negative-p' + str(i) + '-2.png', 0))

        desc1 = extract(img1, keypoints1)
        desc2 = extract(img2, keypoints2)

        y_true.append(1)
        y_scores.append(compare(desc1[1], desc2[1]))

    return y_true, y_scores

if __name__ == "__main__":
    keypoints_own = [[32, 32], [20, 40], [25, 42], [40, 40]]
    print("ROR")
    # y_true, y_scores = test(bright.extract_bright_and_hist, bright.distance_histogram, keypoints_own, keypoints_own)
    y_true, y_scores = test(bright.extract_bright_and_hist, bright.distance_bright, keypoints_own, keypoints_own)
    print(roc_auc_score(y_true, y_scores))
