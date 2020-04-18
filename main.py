#!/usr/bin/env python3

import time
import toml
import click
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from typing import Any, List, Dict
from click import clear, echo, style, secho

from utils import (
    get_image_data,
    export_image,
    select_channel,
    histogram,
    convolve,
    gaussian_kernel,
    sobel_filters,
    non_max_suppression,
    threshold,
    hysteresis,
    find_middle_hist,
    k_means,
)

conf: Dict[str, Any] = {}


def erode(img_arr: np.array, win: int = 1) -> np.array:
    """

    erodes 2D numpy array holding a binary image

    """

    r = np.zeros(img_arr.shape)
    [yy, xx] = np.where(img_arr > 0)

    # prepare neighborhoods
    off = np.tile(range(-win, win + 1), (2 * win + 1, 1))
    x_off = off.flatten()
    y_off = off.T.flatten()

    # duplicate each neighborhood element for each index
    n = len(xx.flatten())
    x_off = np.tile(x_off, (n, 1)).flatten()
    y_off = np.tile(y_off, (n, 1)).flatten()

    # round out offset
    ind = np.sqrt(x_off ** 2 + y_off ** 2) > win
    x_off[ind] = 0
    y_off[ind] = 0

    # duplicate each index for each neighborhood element
    xx = np.tile(xx, ((2 * win + 1) ** 2))
    yy = np.tile(yy, ((2 * win + 1) ** 2))

    nx = xx + x_off
    ny = yy + y_off

    # bounds checking
    ny[ny < 0] = 0
    ny[ny > img_arr.shape[0] - 1] = img_arr.shape[0] - 1
    nx[nx < 0] = 0
    nx[nx > img_arr.shape[1] - 1] = img_arr.shape[1] - 1

    r[ny, nx] = 255

    return r


def dilate(img_arr: np.array, win: int = 1) -> np.array:

    inverted_img = np.invert(img_arr)
    eroded_inverse = erode(inverted_img, win).astype(np.uint8)
    eroded_img = np.invert(eroded_inverse)

    return eroded_img


def histogram_thresholding(img_arr: np.array) -> np.array:

    hist = histogram(img_arr)

    middle = find_middle_hist(hist)

    img_copy = img_arr.copy()
    img_copy[img_copy > middle] = 255
    img_copy[img_copy < middle] = 0

    img_copy = img_copy.astype(np.uint8)

    return img_copy.reshape(img_arr.shape)


def histogram_clustering(img_arr: np.array) -> np.array:
    img_hist = histogram(img_arr)
    out = k_means(img_hist, 2)

    diff = abs(out[1] - out[0])

    img_copy = img_arr.copy()
    img_copy[img_copy > diff] = 255
    img_copy[img_copy < diff] = 0

    img_copy = img_copy.astype(np.uint8)

    return img_copy.reshape(img_arr.shape)


def canny_edge_detection(img_arr: np.array) -> np.array:

    guass = gaussian_kernel(5)

    blurred_image = convolve(img_arr, guass)

    sobel, theta = sobel_filters(blurred_image)

    suppresion = non_max_suppression(sobel, theta)

    threshold_image, weak, strong = threshold(suppresion)

    canny_image = hysteresis(threshold_image, weak, strong)

    return canny_image


def apply_operations(file: Path) -> str:
    """
    Image segmentation–requirement for the project part 2:
    1. Implement one selected edge detection algorithm.
    2. Implement dilation and erosion operators.
    3. Apply segmentation into two groups –foreground (cells) and background (everything else).
    4. Implement two segmentation techniques (they must be implemented by you, not API calls):
        + histogram thresholding
        + histogram clustering (basic approach using two clusters and k-means)
    5. Present example results before and after edge detection
        / dilation
        / erosion
        / segmentation for each respective class of cells (seven in total)
    """

    try:
        img = get_image_data(file)
        img = select_channel(img, conf["COLOR_CHANNEL"])

        # Edge detection
        edges = canny_edge_detection(img)

        # Histogram Clustering Segmentation
        segmented_clustering = histogram_clustering(img)

        # Histogram Thresholding Segmentation
        segmented_thresholding = histogram_thresholding(img)

        # Dilation
        dilated = dilate(segmented_thresholding)

        # Erosion
        eroded = erode(segmented_thresholding)

        export_image(edges, f"edges_{file.stem}", conf)
        export_image(segmented_clustering, f"seg_clusting_{file.stem}", conf)
        export_image(segmented_thresholding, f"seg_thresholding_{file.stem}", conf)
        export_image(dilated, f"dilated_{file.stem}", conf)
        export_image(eroded, f"eroded_{file.stem}", conf)
    except Exception as e:
        return style(f"[ERROR] {file.stem} has an issue: {e}", fg="red")

    return style(f"{f'[INFO:{file.stem}]':15}", fg="green")


def parallel_operations(files: List[Path]):
    """
    Batch operates on a set of images in a multiprocess pool
    """

    echo(
        style("[INFO] ", fg="green")
        + f"initilizing process pool (number of processes: {conf['NUM_OF_PROCESSES']})"
    )
    echo(style("[INFO] ", fg="green") + "compiling...")
    with Pool(conf["NUM_OF_PROCESSES"]) as p:
        with tqdm(total=len(files)) as pbar:
            for res in tqdm(p.imap(apply_operations, files)):
                pbar.write(res + f" finished...")
                pbar.update()


@click.command()
@click.option(
    "config_location",
    "-c",
    "--config",
    envvar="CMSC630_CONFIG",
    type=click.Path(exists=True),
    default="config.toml",
    show_default=True,
)
def main(config_location: str):
    global conf
    conf = toml.load(config_location)

    clear()

    base_path: Path = Path(conf["DATA_DIR"])

    files: List = list(base_path.glob(f"*{conf['FILE_EXTENSION']}"))
    echo(
        style("[INFO] ", fg="green")
        + f"image directory: {str(base_path)}; {len(files)} images found"
    )

    Path(conf["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    # [!!!] Only for development
    # DATA_SUBSET = 1
    # files = files[:DATA_SUBSET]

    t0 = time.time()
    parallel_operations(files)
    t_delta = time.time() - t0

    print()
    secho(f"Total time: {t_delta:.2f} s", fg="green")


if __name__ == "__main__":
    main()
