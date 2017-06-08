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
    r1 = brief.compare(desc0.brief, desc1.brief)
    r2 = distance_bright(desc0.bright, desc1.bright)
    r3 = distance_histogram(desc0.bright, desc1.bright)
    r4 = grad.compare(desc0.gradient, desc1.gradient)
    result = (2 * r1 + .5 * r2 + 1.5 * r3 + r4) / 5.
    if result < 0.05:
        return 0.
    elif result > 0.95:
        return 1.
    return result

