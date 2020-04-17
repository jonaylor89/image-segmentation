#!/usr/bin/env python3

import time
import toml
import click
import numpy as np
from numba import njit
from pathlib import Path
from typing import Any, List, Dict
from click import clear, echo, style, secho

from utils import (
    get_image_data,
    export_image,
    select_channel,
    convolve,
    gaussian_kernel,
    sobel_filters,
    non_max_suppression,
    threshold,
    hysteresis,
    k_means,
)

conf: Dict[str, Any] = {}


@njit(fastmath=True)
def histogram(img_array: np.array) -> np.array:
    """
    >> h=zeros(256,1);              OR    >> h=zeros(256,1);
    >> for l = 0 : 255                    >> for l = 0 : 255
         for i = 1 : N                          h(l +1)=sum(sum(A == l));
            for j = 1 : M                    end
                if (A(i,j) == l)          >> bar(0:255,h);
                    h(l +1) = h(l +1) +1;
                end
            end
        end
    end
    >> bar(0:255,h);
    """

    # Create blank histogram
    hist: np.array = np.zeros(256)

    # Get size of pixel array
    image_size: int = len(img_array)

    for pixel_value in range(256):
        for i in range(image_size):

            # Loop through pixels to calculate histogram
            if img_array.flat[i] == pixel_value:
                hist[pixel_value] += 1

    return hist


def morph_dilation(img_arr: np.array) -> np.array:
    return np.zeros(5)


def morph_erosion(img_arr: np.array) -> np.array:
    return np.zeros(5)


def histogram_thresholding(img_arr: np.array) -> np.array:
    return np.zeros(5)


def histogram_clustering(img_arr: np.array) -> np.array:
    img_hist = histogram(img_arr)
    out = k_means(img_hist, 2)

    diff = +(out[1] - out[1])

    img_copy = np.array([0 if v > diff else 255 for v in img_arr.flat])

    img_copy = img_copy.astype(np.uint8)

    return img_copy.reshape(img_arr.shape)


def canny_edge_detection(img_array: np.array) -> np.array:

    guass = gaussian_kernel(5)

    blurred_image = convolve(img_array, guass)

    sobel, theta = sobel_filters(blurred_image)

    suppresion = non_max_suppression(sobel, theta)

    threshold_image, weak, strong = threshold(suppresion)

    canny_image = hysteresis(threshold_image, weak, strong)

    return canny_image


def apply_operations(files: List[Path]) -> None:
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

    for file in files:
        print(f"operating on {file.stem}")

        img = get_image_data(file)
        img = select_channel(img, conf["COLOR_CHANNEL"])

        # Edge detection
        edges = canny_edge_detection(img)

        # Histogram Clustering Segmentation
        segmented_clustering = histogram_clustering(img)

        # Histogram Thresholding Segmentation
        # segmented_thresholding = histogram_thresholding(img)

        # Dilation
        # dilated = morph_dilation(img)

        # Erosion
        # eroded = morph_erosion(img)

        export_image(edges, f"edges_{file.stem}", conf)
        export_image(segmented_clustering, f"seg_clusting_{file.stem}", conf)
        # export_image(segmented_thresholding, f"seg_thresholding_{file.stem}.BMP")
        # export_image(dilated, f"dilated_{file.stem}.BMP")
        # export_image(eroded, f"eroded_{file.stem}.BMP")



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
    DATA_SUBSET = 1
    files = files[:DATA_SUBSET]

    t0 = time.time()
    apply_operations(files)
    t_delta = time.time() - t0

    print()
    secho(f"Total time: {t_delta:.2f} s", fg="green")


if __name__ == "__main__":
    main()
