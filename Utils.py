import math


def argmax(l):
    return max(enumerate(l), key=lambda x: x[1])[0]


def argmin(l):
    return min(enumerate(l), key=lambda x: x[1])[0]


def calc_uncertainty(predictions, alpha=0.5, method="confidence"):
    p = predictions[1]
    p = math.pow(p, -math.log(2, alpha))
    uncert = 1
    if method is "confidence":
        uncert = 1 - abs(1 - 2 * p)
    elif method is "variance":
        uncert = p * (1 - p)
    elif method is "entropy":
        uncert = p * math.log(p) + (1 - p) * math.log(1 - p)
    else:
        raise ValueError("Incorrect method")
    return uncert
