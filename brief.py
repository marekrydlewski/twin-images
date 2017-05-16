from skimage import data
from skimage import transform as tf
from skimage.feature import (match_descriptors, corner_peaks, corner_harris,
                             plot_matches, BRIEF)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian
from skimage.util.dtype import img_as_bool


class ModBrief:

    def __init__(self, descriptor_size=256, patch_size=63, sigma=1.):
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
        gaussian_val = 1. / 5. * self.patch_size
        samples = np.random.normal(0, gaussian_val, self.descriptor_size * 8)
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
                temp = 0
                if pointX > pointY:
                    temp = True
                else:
                    temp = False
                self.descriptors[y, x] = temp
        print("dupa")



    def compare(self, desc0, desc1):
        matches = match_descriptors(desc0, desc1, cross_check=True)
        return 1. - 2. * float(matches.shape[0]) / float(desc0.shape[0] + desc1.shape[0])



img1 = rgb2gray(data.rocket())

tform = tf.AffineTransform(scale=(1.8, 1.2), translation=(0, -100))
img2 = tf.warp(img1, tform)
img3 = tf.rotate(img1, 55)

# img2 = rgb2gray(data.hubble_deep_field())

keypoints1 = corner_peaks(corner_harris(img1), min_distance=5)
keypoints2 = corner_peaks(corner_harris(img2), min_distance=5)
keypoints3 = corner_peaks(corner_harris(img3), min_distance=5)


extractor = ModBrief()
#extractor = BRIEF()


extractor.extract(img1, keypoints1)
keypoints1 = keypoints1[extractor.mask]
descriptors1 = extractor.descriptors

extractor.extract(img2, keypoints2)
keypoints2 = keypoints2[extractor.mask]
descriptors2 = extractor.descriptors

extractor.extract(img3, keypoints3)
keypoints3 = keypoints3[extractor.mask]
descriptors3 = extractor.descriptors

matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
matches13 = match_descriptors(descriptors1, descriptors3, cross_check=True)


print(descriptors1.shape[0])
print(descriptors2.shape[0])
print(descriptors3.shape[0])
print(matches12.shape[0])
print(matches13.shape[0])

# matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True, metric='sqeuclidean')
# matches13 = match_descriptors(descriptors1, descriptors3, cross_check=True, metric='hamming')

fig, ax = plt.subplots(nrows=2, ncols=1)

plt.gray()

plot_matches(ax[0], img1, img2, keypoints1, keypoints2, matches12, matches_color=(0.1, 0.3, 0.8))
ax[0].axis('off')
ax[0].set_title("Original Image vs. Transformed Image")


plot_matches(ax[1], img1, img3, keypoints1, keypoints3, matches13, matches_color=(0.1, 0.3, 0.8))
ax[1].axis('off')
ax[1].set_title("Original Image vs. Transformed Image")

print(extractor.compare(descriptors1, descriptors1))
print(extractor.compare(descriptors1, descriptors3))
plt.show()