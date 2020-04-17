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

@njit(fastmath=True)
def apply_median_filter(img_array: np.array, img_filter: np.array) -> np.array:
    """
    Applies a linear filter to a copy of an image based on filter weights
    """

    rows, cols = img_array.shape
    height, width = img_filter.shape

    pixel_values = np.zeros(img_filter.size ** 2)
    output = np.zeros((rows - height + 1, cols - width + 1))

    for rr in range(rows - height + 1):
        for cc in range(cols - width + 1):

            p = 0
            for hh in range(height):
                for ww in range(width):

                    pixel_values[p] = img_array[hh][ww]
                    p += 1

            # Sort the array of pixels inplace
            pixel_values.sort()

            # Assign the median pixel value to the filtered image.
            output[rr][cc] = pixel_values[p // 2]

    return output


def median_filter(
    img_array: np.array, mask_size: int, weights: List[List[int]]
) -> np.array:
    """
    median_filter also uses a kernel or matrix of weights,
    given as a two dimensional List, and applies that kernel to a copy of an image.
    The median filter has the added effect of taking the median pixel value of a given neighborhood,
    of which the size of that neighborhood is specified by the size of the kernel,
    and assigning that value to the pixel in question at that moment.
    """

    filter = np.array(weights)
    median = apply_median_filter(img_array, filter)

    return median

