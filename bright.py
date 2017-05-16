from skimage import data
from skimage import transform as tf
from skimage.color import rgb2gray
import numpy as np
import cv2
from skimage.filters import gaussian


def extract_bright(image, keypoints):
    window = int(64 / 2)
    sigma = 1.
    image = gaussian(image, sigma=sigma)
    descriptors = []
    for keypoint in keypoints:
        subimage = image[max(keypoint[0] - window, 0): min(keypoint[0] + window, image.shape[0])][max(keypoint[1] - window, 0): min(keypoint[1] + window, image.shape[1])]
        squeezed = np.squeeze(subimage)
        descriptors.append((np.std(subimage), np.average(subimage)))
    return descriptors


def distance_bright(descriptor1, descriptor2):
    return abs(descriptor1[0] - descriptor2[0]) + abs(descriptor1[1] - descriptor2[1])


# same object, different images
img1 = rgb2gray(cv2.imread('samples/bikes/00004.png', 0))
img2 = rgb2gray(cv2.imread('samples/bikes/00005.png', 0))

# different objects
img3 = rgb2gray(cv2.imread('samples/bikes/00030.png', 0))
img4 = rgb2gray(cv2.imread('samples/bikes/00079.png', 0))

keypoints_own = [[32, 32], [20, 40], [25, 42]]

desc1 = extract_bright(img1, keypoints_own)
desc2 = extract_bright(img2, keypoints_own)
score = distance_bright(desc1[0], desc2[0])
score1 = distance_bright(desc1[1], desc2[1])
score2 = distance_bright(desc1[2], desc2[2])
print(score)
print(score1)
print(score2)

desc3 = extract_bright(img3, keypoints_own)
desc4 = extract_bright(img4, keypoints_own)
score3 = distance_bright(desc1[0], desc4[0])
score4 = distance_bright(desc1[1], desc4[1])
score5 = distance_bright(desc1[2], desc4[2])
print(score3)
print(score4)
print(score5)

