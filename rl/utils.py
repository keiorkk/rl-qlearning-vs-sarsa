import numpy as np
from typing import Iterable, List

def set_seed(seed: int = 42):
    np.random.seed(seed)

def linear_epsilon(start: float, end: float, total_steps: int):
    # yields epsilon for step t in [0, total_steps-1]
    for t in range(total_steps):
        if total_steps <= 1:
            yield end
        else:
            yield start + (end - start) * (t / (total_steps - 1))

def moving_average(x: Iterable[float], window: int = 50) -> list:
    x = list(x)
    if len(x) == 0:
        return []
    w = min(window, max(1, len(x)))
    out = []
    cumsum = 0.0
    for i, v in enumerate(x):
        cumsum += v
        if i >= w:
            cumsum -= x[i - w]
            out.append(cumsum / w)
        else:
            out.append(cumsum / (i + 1))
    return out
