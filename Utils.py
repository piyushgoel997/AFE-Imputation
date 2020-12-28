def argmax(l):
    return max(enumerate(l), key=lambda x: x[1])[0]


def argmin(l):
    return min(enumerate(l), key=lambda x: x[1])[0]


def calc_uncertainty(predictions):
    return 1 - max(predictions)
