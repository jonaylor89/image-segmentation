#!/usr/bin/env python3

import time
import toml
import click
from math import sqrt
import numpy as np
from numba import njit
from PIL import Image
from pathlib import Path
from typing import List, Optional
from click import clear, echo, style, secho

conf: Optional[dict] = None


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


def morph_dilation() -> np.array:
    return np.zeros(5)


def morph_erosion() -> np.array:
    return np.zeros(5)


@njit(parallel=True)
def kmeans(A: np.array, numCenter: int, numIter: int, size: int, features: int) -> np.array:
    # https://github.com/numba/numba/blob/master/examples/k-means/k-means_numba.py
    centroids = np.random.ranf((numCenter, features))

    for l in range(numIter):
        dist = np.array(
            [
                [
                    sqrt(np.sum((A[i, :] - centroids[j, :]) ** 2))
                    for j in range(numCenter)
                ]
                for i in range(size)
            ]
        )
        labels = np.array([dist[i, :].argmin() for i in range(size)])

        centroids = np.array(
            [
                [np.sum(A[labels == i, j]) / np.sum(labels == i) for j in range(features)]
                for i in range(numCenter)
            ]
        )

    return centroids


def histogram_thresholding() -> np.array:
    return np.zeros()


def histogram_clustering() -> np.array:
    return np.zeros(5)


def canny_edge_detection() -> np.array:
    # https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
    return np.zeros(5)


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
        print(file.stem)

        # Edge detection
        edges = canny_edge_detection()

        # Dilation
        dilated = morph_dilation()

        # Erosion
        eroded = morph_erosion()

        # Histogram Clustering Segmentation
        segmented_clustering = histogram_clustering()

        # Histogram Thresholding Segmentation
        segmented_thresholding = histogram_thresholding()

        export_image(edges, f"edges_{file.stem}.BMP")
        export_image(dilated, f"dilated_{file.stem}.BMP")
        export_image(eroded, f"eroded_{file.stem}.BMP")
        export_image(segmented_clustering, f"seg_clusting_{file.stem}.BMP")
        export_image(segmented_thresholding, f"seg_thresholding_{file.stem}.BMP")


def select_channel(img_array: np.array, color: str = "red") -> np.array:
    """
    select_channel isolates a color channel from a RGB image represented as a numpy array.
    """
    if color == "red":
        return img_array[:, :, 0]

    elif color == "green":
        return img_array[:, :, 1]

    elif color == "blue":
        return img_array[:, :, 2]


def export_image(img_arr: np.array, filename: str) -> None:
    """
    Exports a numpy array as a grey scale bmp image
    """
    img = Image.fromarray(img_arr)
    img = img.convert("L")
    img.save(conf["OUTPUT_DIR"] + filename + conf["FILE_EXTENSION"])


def get_image_data(filename: Path) -> np.array:
    """
    Converts a bmp image to a numpy array
    """

    with Image.open(filename) as img:
        return np.array(img)


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
    DATA_SUBSET = 5
    files = files[:DATA_SUBSET]

    t0 = time.time()
    apply_operations(files)
    t_delta = time.time() - t0

    print()
    secho(f"Total time: {t_delta:.2f} s", fg="green")


if __name__ == "__main__":
    main()
