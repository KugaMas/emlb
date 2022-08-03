import time
import numpy as np
from scipy.ndimage import median_filter


def hotpixel_removal(ev, size, threshold=100):
    """
    Exclude all possible hotpixel through event distribution histogram
    Args:
        ev: the event data
        size: the DVS sensor size
        threshold: find pixels whose distribution gradient is higher than the threshold
    return:
        ev: the event data
    """
    _bins, _range = size, [[0, size[0]], [0, size[1]]]
    hist, *_ = np.histogram2d(ev[:, 1], ev[:, 2], bins=_bins, range=_range)
    seek = np.where((hist - median_filter(hist, 3)).flatten() >= threshold)

    idx, *_ = ~np.in1d(np.ravel_multi_index(ev[:, 1:3].T, size), seek)
    
    return ev[idx]


def polarity_removal(ev, size):
    """

    :return:
    """
    ev[:, 3] = 0
    return ev

