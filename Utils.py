import math


def argmax(l):
    return max(enumerate(l), key=lambda x: x[1])[0]


def argmin(l):
    return min(enumerate(l), key=lambda x: x[1])[0]


def calc_uncertainty(predictions, alpha=0.5, method="confidence"):
    p = predictions[1]
    if p == 0 or p == 1:
        return 0
    p = math.pow(p, -math.log(2, alpha))
    if method == "confidence":
        uncert = 1 - abs(1 - 2 * p)
    elif method == "variance":
        uncert = p * (1 - p)
    elif method == "entropy":
        uncert = p * math.log(p) + (1 - p) * math.log(1 - p)
    else:
        print("The method name is", method)
        raise ValueError("Incorrect uncertainty method")
    return uncert
