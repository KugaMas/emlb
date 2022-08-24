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


def count_distribution(ev, size, use_polarity=True):
    bins_, range_  = [size[0], size[1]], [[0, size[0]], [0, size[1]]]

    if use_polarity:
        weights = (-1) ** (1 + ev[:, 3].astype(np.float_))
    else:
        weights = (+1) ** (0 + ev[:, 3].astype(np.float_))

    counts, *_ = np.histogram2d(ev[:, 1], ev[:, 2], weights=weights, bins=bins_, range=range_)

    return counts


def pack_along_timestamp(ev, size, duration):
    duration = int(duration)
    ts_refer = np.arange(ev[0, 0], ev[-1, 0], duration)
    refer_index = np.searchsorted(ev[:, 0], ts_refer)[1:]

    packets = [ei for ei in np.split(ev, refer_index)]
    return packets


def projection_image(ev, size, max_count=1, flip=True):
    cnt = count_distribution(ev, size)
    cnt = np.clip(np.abs(cnt), 0, max_count).astype(np.int_)

    color = np.linspace(0, 255, max_count + 1)

    img = np.ones((*size, 3)) * 255
    img[ev[:, 1], ev[:, 2], 1 - ev[:, 3]] = color[-1 - cnt[ev[:, 1], ev[:, 2]]]
    img[ev[:, 1], ev[:, 2], 2 - ev[:, 3]] = color[-1 - cnt[ev[:, 1], ev[:, 2]]]

    if flip:
        img = np.flip(np.rot90(img, 1), axis=0).astype(np.uint8)

    return img


def calc_event_structural_ratio(ev, size, count=30000, refN=20000):
    score = np.zeros(int(len(ev)/count) - 1)

    for i in range(0, len(score)):
        st_idx = i * count
        ed_idx = st_idx + count
        packet = ev[st_idx:ed_idx]

        cnt = count_distribution(packet, size, use_polarity=False)

        N = cnt.sum()
        L = cnt.size - ((1 - refN/N) ** cnt).sum()

        score[i] = (cnt * (cnt-1) / (N + np.spacing(1)) / (N - 1 + np.spacing(1))).sum() * L

    return score.mean()

    # cnt = [count_distribution(ev[k][idx], size, use_polarity=False) for k in range(0, len(ev))]

    # N = [c.sum() for c in cnt]
    # L = [c.size - ((1 - refN/c.sum())**c).sum() for c in cnt]
    # ecr = [(c * (c-1) / (n + np.spacing(1)) / (n - 1 + np.spacing(1))).sum() * l for c, n, l in zip(cnt, N, L)]

    # # N = [c.sum() for c in cnt]
    # # connectivity = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    # # L = [(cv.filter2D((c != 0) * 1., -1, connectivity)[c != 0] > 0).sum() for c in cnt]
    # # ecr = [(c * (c - 1) / n / (n - 1)).sum() * l for c, n, l in zip(cnt, N, L)]

    # ecr = np.round(ecr, 2)
    # ecr[np.array(N) < refN] = np.nan

    # return np.round(ecr, 2)

