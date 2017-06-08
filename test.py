from sklearn.metrics import roc_auc_score
from skimage.color import rgb2gray
import cv2
import numpy as np
import descriptor


def test(extract, compare, keypoints1, keypoints2):
    y_true = []
    y_scores = []
    for i in range(1, 350):
        img1 = rgb2gray(cv2.imread('test/positive-p' + str(i) + '-1.png', 0))
        img2 = rgb2gray(cv2.imread('test/positive-p' + str(i) + '-2.png', 0))

        desc1 = extract(img1, keypoints1)
        desc2 = extract(img2, keypoints2)

        y_true.append(0)
        y_scores.append(compare(desc1[0], desc2[0]))

    for i in range(1, 350):
        img1 = rgb2gray(cv2.imread('test/negative-p' + str(i) + '-1.png', 0))
        img2 = rgb2gray(cv2.imread('test/negative-p' + str(i) + '-2.png', 0))

        desc1 = extract(img1, keypoints1)
        desc2 = extract(img2, keypoints2)

        y_true.append(1)
        y_scores.append(compare(desc1[0], desc2[0]))

    return y_true, y_scores

x_large_image = descriptor.extract(rgb2gray(cv2.imread('large/negative-p0-1.png', 0)), [(70, 31), (92, 31)])
y_large_image = descriptor.extract(rgb2gray(cv2.imread('large/negative-p0-2.png', 0)), [(70, 31), (92, 31)])
dis_large = descriptor.distance(x_large_image[0], y_large_image[0])

a, b = test(descriptor.extract, descriptor.distance, np.array([[32, 32], [12, 15], [28, 27]]), np.array([[32, 32], [12, 15], [28, 27]]))

for threshold in range(10, 90, 5):
    temp = [1 if x < threshold / 100 else 0 for x in b[:350]]
    temp2 = [0 if x < threshold / 100 else 1 for x in b[350:]]
    asd = temp.count(1) / 350
    qwe = temp2.count(1) / 350
    print(threshold.__str__() + " : pos =  " + asd.__str__() + "; neg = " + qwe.__str__())
print(roc_auc_score(a, b))
