from skimage import data
from skimage import transform as tf
from skimage.feature import (match_descriptors, plot_matches, BRIEF)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import cv2
from enum import Enum

class Direction(Enum):
    LEFT = 0
    LEFT_UP = 1
    UP = 2
    RIGHT_UP = 3
    RIGHT = 4
    RIGHT_DOWN = 5
    DOWN = 6
    LEFT_DOWN = 7

class GradientDesc():
    def __init__(self):
        self.smallBlockSize = 14
        self.maxWindowSize = 64

    def computePoint(self, image, x, y):
        result = np.zeros(8)
        mask = [(-1, 0),
                (-1, 1), (0, 1), (1, 1),
                (1, 0),
                (1, -1), (0, -1), (-1, -1)
                ]

        for a in range(8):
            partial = float(image[x + mask[a][0], y + mask[a][1]]) - float(image[x][y])
            if partial < 0:
                result[(a + 4) % 8] += -partial
            else:
                result[a] += partial

        return self.normaliseHistogram(result)
        # for a in range(4):
        #     result[a] -= min(result[a], result[a+4])
        #     result[a+4] -= min(result[a], result[a+4])
        #
        # new_result = np.zeros(8)
        # for a in range(8):
        #     new_result[a] = (result[a - 1] + result[a] + result[(a+1)%8])/3.
        #
        # return np.argmax(new_result)

    def normaliseHistogram(self, histogram):
        for a in range(4):
            histogram[a] -= min(histogram[a], histogram[a+4])
            histogram[a+4] -= min(histogram[a], histogram[a+4])
        new_result = np.zeros(8)
        for a in range(8):
            new_result[a] = (histogram[a - 1] + histogram[a] + histogram[(a+1)%8])/3.
        return np.argmax(new_result)

    def extract(self, image, points):
        descriptor = []
        for x in points:
            descriptor.append(self.extractOne(image, x))
        return np.array(descriptor)

    def extractOne(self, image, point, safeMode=True):
        if safeMode:
            distance = 2 * self.smallBlockSize + 1
            maxX = image.shape[0]
            maxY = image.shape[1]
            if point[0] - distance < 0 or point[0] + distance >= maxX or point[1] - distance < 0 or point[1] + distance >= maxY:
                return []

        new_points = [ [point[0] - 28, point[1] - 7],
                       [point[0] - 14, point[1]],
                       [point[0] - 7, point[1] + 14],
                        point,
                       [point[0] + 14, point[1] - 7],
                       [point[0], point[1] - 14],
                       [point[0] - 7, point[1] - 28],
                       [point[0] - 14, point[1] - 14]
                       ]

        descriptor = np.zeros(8)
        for p in range(8):
            table = np.zeros(8)
            for x in range(14):
                for y in range(14):
                    table[self.computePoint(image, new_points[p][0]+x, new_points[p][1]+y)] += 1
            descriptor[p] = self.normaliseHistogram(table)

        return descriptor

    def compareDescriptors(self, desc0, desc1):
        match = np.zeros(8)
        for x in range(8):
            for a in range(8):
                match[x] += abs(desc0[a] - desc1[(x + a) % 8])
        temp = float(min(match))/float(max(desc0.sum(), desc1.sum()))
        return temp


if __name__ == "__main__":
    # 'samples/bikes/00004.png'
    # 'samples/raw/bikes/img1.ppm'
    img1 = rgb2gray(cv2.imread('samples/bikes/00000.png', 0))

    #tform = tf.AffineTransform(scale=(1.8, 1.2), translation=(0, -100))
    #img2 = tf.warp(img1, tform)
    img2 = rgb2gray(cv2.imread('samples/bikes/00039.png', 0))
    img3 = rgb2gray(cv2.imread('samples/bark/00000.png', 0))
    #img3 = tf.rotate(img2, 25)

    # img2 = rgb2gray(data.hubble_deep_field())

    #keypoints1 = corner_peaks(corner_harris(img1), min_distance=5)
    #keypoints2 = corner_peaks(corner_harris(img2), min_distance=5)
    #keypoints3 = corner_peaks(corner_harris(img3), min_distance=5)

    keypoints1 = np.array([[32, 32]])
    keypoints2 = np.array([[32, 32]])
    keypoints3 = np.array([[32, 32]])


    extractor = GradientDesc()
    #extractor = BRIEF()


    descriptors1 = extractor.extract(img1, keypoints1)
    descriptors2 = extractor.extract(img2, keypoints2)
    descriptors3 = extractor.extract(img3, keypoints3)

    norm1 = extractor.normaliseHistogram(descriptors1[0])
    norm2 = extractor.normaliseHistogram(descriptors2[0])
    norm3 = extractor.normaliseHistogram(descriptors3[0])

    t11 = extractor.compareDescriptors(descriptors1[0], descriptors1[0])
    t12 = extractor.compareDescriptors(descriptors1[0], descriptors2[0])
    t13 = extractor.compareDescriptors(descriptors1[0], descriptors3[0])
    t23 = extractor.compareDescriptors(descriptors3[0], descriptors2[0])
    print(t11)
    print(t12)
    print(t13)
    print(t23)
    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
    matches13 = match_descriptors(descriptors1, descriptors3, cross_check=True)


    fig, ax = plt.subplots(nrows=2, ncols=1)

    plt.gray()

    plot_matches(ax[0], img1, img2, keypoints1, keypoints2, matches12, matches_color=(0.1, 0.3, 0.8))
    ax[0].axis('off')
    ax[0].set_title("Original Image vs. Transformed Image")


    plot_matches(ax[1], img1, img3, keypoints1, keypoints3, matches13, matches_color=(0.1, 0.3, 0.8))
    ax[1].axis('off')
    ax[1].set_title("Original Image vs. Transformed Image")

    plt.show()