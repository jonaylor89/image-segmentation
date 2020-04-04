#!/usr/bin/env python3
import time
import toml
import click
from pathlib import Path
from click import clear, echo, style, secho


def dilation():
    pass


def erosion():
    pass


def kmeans():
    # https://github.com/numba/numba/blob/master/examples/k-means/k-means_numba.py
    pass


def histogram_thresholding():
    pass


def histogram_clustering():
    pass


def canny_edge_detection():
    # https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
    pass


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
def main(config_location):
    conf = toml.load(config_location)

    clear()

    base_path = Path(conf["DATA_DIR"])

    files = list(base_path.glob("*.BMP"))
    echo(
        style("[INFO] ", fg="green")
        + f"image directory: {str(base_path)}; {len(files)} images found"
    )

    Path(conf["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # [!!!] Only for development
    # DATA_SUBSET = 5; files = files[:DATA_SUBSET]

    t_delta = time.time() - t0

    print()
    secho(f"Total time: {t_delta:.2f} s", fg="green")


if __name__ == "__main__":
    main()
