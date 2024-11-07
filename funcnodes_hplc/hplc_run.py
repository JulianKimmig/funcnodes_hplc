from typing import Optional
import numpy as np
from scipy.signal import find_peaks


def estimate_t0(x: np.ndarray, y: np.ndarray, maximum=4, minimum=0) -> Optional[float]:
    t0max = min(maximum, (x.max() - x.min()) / 10)
    t0maxidx = np.argmin(np.abs(x - t0max))
    yr = y[:t0maxidx]

    # get fist local minima after first local maxima
    # get first local maxima
    p = find_peaks(yr, height=(y.max() - y.min()) * 0.001)[0]
    if len(p) == 0:
        return minimum
    p0 = p[0]

    # get first local minima

    p = find_peaks(-yr[p0 + 1 :])[0]
    if len(p) == 0:
        return minimum
    p = p[0] + p0

    return x[p]
