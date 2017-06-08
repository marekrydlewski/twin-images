from skimage.feature import match_descriptors
import numpy as np
from skimage.filters import gaussian
from scipy.spatial.distance import hamming


class ModBrief:

    def __init__(self, descriptor_size=128, patch_size=32, sigma=1.):
        self.descriptor_size = descriptor_size
        self.patch_size = patch_size
        self.sigma = sigma
        self.mask = None
        self.descriptors = None

    def extract(self, image, keypoints, safemode=True):
        image = gaussian(image, self.sigma)
        #
        # get random (normal distribution) relative points
        #
        random = np.random.RandomState()
        random.seed(1)
        samples = (self.patch_size / 5.) * random.randn(self.descriptor_size * 8)
        samples = np.array(samples, dtype=np.int32)
        samples = samples[(samples < (self.patch_size // 2)) & (samples > - (self.patch_size - 2) // 2)]
        p1 = samples[: self.descriptor_size * 2].reshape(self.descriptor_size, 2)
        p2 = samples[self.descriptor_size*2:self.descriptor_size*4].reshape(self.descriptor_size, 2)
        #
        # safemode disable points too close to border
        #
        self.mask = np.ones((keypoints.shape[0]), dtype=bool)
        if safemode:
            distance = self.patch_size // 2
            for point in range(keypoints.shape[0]):
                if keypoints[point, 0] < distance or keypoints[point, 0] > image.shape[0] - distance or keypoints[point, 1] < distance or keypoints[point, 1] > image.shape[1] - distance:
                    self.mask[point] = False
            keypoints = np.array(keypoints[self.mask,:], dtype=np.int32,)#1 order='C', copy=False)
        self.descriptors = np.zeros((keypoints.shape[0], self.descriptor_size), dtype=float, order='C')
        #
        # compute distance function
        #
        # image = np.array(image, dtype=np.int32)
        for x in range(p1.shape[0]):
            pr0 = p1[x, 0]
            pc0 = p1[x, 1]
            pr1 = p2[x, 0]
            pc1 = p2[x, 1]
            for y in range(keypoints.shape[0]):
                kr = keypoints[y, 0]
                kc = keypoints[y, 1]
                pointX = image[kr + pr0, kc + pc0]
                pointY = image[kr + pr1, kc + pc1]
                temp = None
                if pointX > pointY:
                    temp = True
                else:
                    temp = False
                self.descriptors[y, x] = temp
        return self.descriptors


    def compare(self, desc0, desc1):
        return hamming(desc0, desc1)
        # if desc0.shape[0] == 1 and desc1.shape[0] == 1:
        #     return hamming(desc0, desc1)
        # matches = match_descriptors(desc0, desc1, cross_check=True)
        # return 1. - 2. * float(matches.shape[0]) / float(desc0.shape[0] + desc1.shape[0])

