from sklearn.metrics import roc_auc_score
from main import *


def test(extract, compare, keypoints1, keypoints2):
    y_true = []
    y_scores = []
    for i in range(1, 350):
        img1 = rgb2gray(cv2.imread('test/positive-p' + str(i) + '-1.png', 0))
        img2 = rgb2gray(cv2.imread('test/positive-p' + str(i) + '-2.png', 0))

        desc1 = extract(img1, keypoints1)
        desc2 = extract(img2, keypoints2)

        y_true.append(0)
        y_scores.append(compare(desc1, desc2))

    for i in range(1, 350):
        img1 = rgb2gray(cv2.imread('test/negative-p' + str(i) + '-1.png', 0))
        img2 = rgb2gray(cv2.imread('test/negative-p' + str(i) + '-2.png', 0))

        desc1 = extract(img1, keypoints1)
        desc2 = extract(img2, keypoints2)

        y_true.append(1)
        y_scores.append(compare(desc1, desc2))

    return y_true, y_scores

a, b = test(extract, distance, [(32,32)], [(32,32)])
print(roc_auc_score(a, b))
