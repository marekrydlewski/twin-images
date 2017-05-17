from skimage import data
from skimage import transform as tf
from skimage.color import rgb2gray
import numpy as np
import cv2
from skimage.filters import gaussian


def extract_bright_and_hist(image, keypoints):
    window = int(64 / 2)
    sigma = 1.
    image = gaussian(image, sigma=sigma)
    descriptors = []
    for keypoint in keypoints:
        subimage = image[max(keypoint[0] - window, 0): min(keypoint[0] + window, image.shape[0])][
                   max(keypoint[1] - window, 0): min(keypoint[1] + window, image.shape[1])]
        # squeezed = np.squeeze(subimage)
        hist, bins = np.histogram(subimage, bins=20, range=(0.0, 1.0))
        descriptors.append((np.std(subimage), np.average(subimage), hist.ravel().astype('float32')))
    return descriptors


def distance_bright(descriptor1, descriptor2):
    return 4 * (abs(descriptor1[0] - descriptor2[0]) +
                2 * abs(descriptor1[1] - descriptor2[1]))


def distance_histogram(descriptor1, descriptor2):
    return 1 - abs(cv2.compareHist(descriptor1[2], descriptor2[2], cv2.HISTCMP_CORREL))

if __name__ == "__main__":
    # same object, different images
    img1 = rgb2gray(cv2.imread('samples/bikes/00216.png', 0))
    img2 = rgb2gray(cv2.imread('samples/bikes/00217.png', 0))

    # different objects
    img3 = rgb2gray(cv2.imread('samples/bikes/00030.png', 0))
    img4 = rgb2gray(cv2.imread('samples/bikes/00079.png', 0))

    keypoints_own = [[32, 32], [20, 40], [25, 42]]

    desc1 = extract_bright_and_hist(img1, keypoints_own)
    desc2 = extract_bright_and_hist(img2, keypoints_own)
    score = distance_bright(desc1[0], desc2[0])
    score1 = distance_bright(desc1[1], desc2[1])
    score2 = distance_bright(desc1[2], desc2[2])
    print(score)
    print(score1)
    print(score2)

    desc3 = extract_bright_and_hist(img3, keypoints_own)
    desc4 = extract_bright_and_hist(img4, keypoints_own)
    score3 = distance_bright(desc1[0], desc4[0])
    score4 = distance_bright(desc1[1], desc4[1])
    score5 = distance_bright(desc1[2], desc4[2])
    print(score3)
    print(score4)
    print(score5)

    score = distance_histogram(desc1[0], desc2[0])
    score1 = distance_histogram(desc1[1], desc2[1])
    score2 = distance_histogram(desc1[2], desc2[2])
    print(score)
    print(score1)
    print(score2)

    score3 = distance_histogram(desc1[0], desc4[0])
    score4 = distance_histogram(desc1[1], desc4[1])
    score5 = distance_histogram(desc1[2], desc4[2])
    print(score3)
    print(score4)
    print(score5)
