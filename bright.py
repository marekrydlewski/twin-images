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
        hist, bins = np.histogram(subimage, bins=20, range=(0.0, 1.0))
        descriptors.append((np.std(subimage), np.average(subimage), hist.ravel().astype('float32')))
    return descriptors


def distance_bright(descriptor1, descriptor2):
    return 4 * (abs(descriptor1[0] - descriptor2[0]) +
                2 * abs(descriptor1[1] - descriptor2[1]))


def distance_histogram(descriptor1, descriptor2):
    return 1 - abs(cv2.compareHist(descriptor1[2], descriptor2[2], cv2.HISTCMP_CORREL))
