from skimage.feature import match_descriptors, plot_matches
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import cv2
from enum import Enum

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

    def compare(self, desc0, desc1):
        match = np.zeros(8)
        for x in range(8):
            for a in range(8):
                match[x] += abs(desc0[a] - desc1[(x + a) % 8])
        temp = float(min(match))/float(max(desc0.sum(), desc1.sum()))
        return temp

