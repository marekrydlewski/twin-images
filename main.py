from brief import ModBrief
from bright import *
from gradient import GradientDesc

class Descriptor:
    def __init__(self):
        self.brief = []
        self.bright = []
        self.gradient = []

def extract(image, keypoints):
    results = Descriptor()
    brief = ModBrief()
    results.brief = brief.extract(image, np.array(keypoints))
    results.bright = extract_bright_and_hist(image, keypoints)
    grad = GradientDesc()
    results.gradient = grad.extract(image, keypoints)
    return results

def distance(desc0, desc1):
    brief = ModBrief()
    grad = GradientDesc()
    r1 = brief.compare(desc0.brief, desc1.brief)
    r2 = distance_bright(desc0.bright[0], desc1.bright[0])
    r3 = distance_histogram(desc0.bright[0], desc1.bright[0])
    r4 = grad.compare(desc0.gradient[0], desc1.gradient[0])
    result = (r1 * 2. + r2 * 2. + r3 * 2. + r4) / 7.
    if result < 0.05:
        return 0.
    elif result > 0.95:
        return 1.
    return result

