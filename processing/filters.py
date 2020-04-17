#!/usr/bin/env python3

import time
import numpy as np
from typing import List
from numba import njit

# timeit: decorator to time functions
def timeit(f, single_time_data):
    def timed(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()

        single_time_data[f.__name__] = (te - ts) * 1000

        return result

    return timed


@njit(fastmath=True)
def mean_square_error(original_img: np.array, quantized_img: np.array) -> int:
    """
    mean_square_error takes two images represented as numpy arrays
    and finds the mean squared error of all of their pixel values.
    """

    mse = (np.square(original_img - quantized_img)).mean()

    return mse


