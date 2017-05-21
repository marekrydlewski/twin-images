from brief import ModBrief
from bright import *
from gradient import GradientDesc

class Descriptor:
    def __init__(self, brief, bright, gradient):
        self.brief = brief
        self.bright = bright
        self.gradient = gradient


def extract(image, keypoints):
    brief = ModBrief()

    briefs = brief.extract(image, np.array(keypoints))

    brights = extract_bright_and_hist(image, keypoints)

    grad = GradientDesc()
    gradients = grad.extract(image, keypoints)

    results = [Descriptor(t[0], t[1], t[2]) for t in zip(briefs, brights, gradients)]
    return results


def distance(desc0, desc1):
    brief = ModBrief()
    grad = GradientDesc()
    r1 = brief.compare(desc0[0].brief, desc1[0].brief)
    r2 = distance_bright(desc0[0].bright, desc1[0].bright)
    r3 = distance_histogram(desc0[0].bright, desc1[0].bright)
    r4 = grad.compare(desc0[0].gradient, desc1[0].gradient)
    result = (2 * r1 + r2 + r3 + r4) / 4.
    if result < 0.05:
        return 0.
    elif result > 0.95:
        return 1.
    return result

