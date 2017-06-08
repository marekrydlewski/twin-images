from skimage.feature import match_descriptors, plot_matches
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean


class GradientDesc():
    def __init__(self):
        self.smallBlockSize = 14
        self.maxWindowSize = 64
        self.patch = self.maxWindowSize // 2 - 1

    def computePoint(self, image, x, y):
        result = np.zeros(8)
        mask = [(-1, 0),
                (-1, 1), (0, 1), (1, 1),
                (1, 0),
                (1, -1), (0, -1), (-1, -1)
                ]
        for a in range(8):
            result[a] = float(image[x + mask[a][0], y + mask[a][1]]) - float(image[x][y])

        return result

    def extract(self, image, points):
        descriptor = []
        for x in points:
            descriptor.append(self.extractOne(image, x))
        return np.array(descriptor)

    def is_feature(self, histogram):
        result = 0
        for x in range(0, 4):
            if np.sign(histogram[x]) == np.sign(histogram[x - 4]):
                result += 1
        return result

    def extractOne(self, image, point, safeMode=True):

        if safeMode:
            distance = 2 * self.smallBlockSize + 1
            maxX = image.shape[0]
            maxY = image.shape[1]
            if point[0] - distance < 0 or point[0] + distance >= maxX or point[1] - distance < 0 or point[
                1] + distance >= maxY:
                return []

        result = []
        for x in range(-self.patch, self.patch):
            for y in range(-self.patch, self.patch):
                new_x = point[0] + x
                new_y = point[0] + y
                result.append(self.is_feature(self.computePoint(image, new_x, new_y)))

        descriptor = np.zeros(5)
        for x in result:
            descriptor[x] += 1

        return descriptor

    def compare(self, desc0, desc1):
        patch = (self.patch * 2) ** 2
        desc0 = [x / patch for x in desc0]
        desc1 = [x / patch for x in desc1]
        return euclidean(desc0, desc1)
