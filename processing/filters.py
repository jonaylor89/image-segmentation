#!/usr/bin/env python3

import time
import toml
import click
import matplotlib
import numpy as np
from typing import List, Tuple
from sys import platform
from collections import Counter
from functools import partial
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from click import clear, echo, style, secho
from multiprocessing import Pool, Queue, Manager
from numba import njit, jit
from matplotlib import pyplot as plt

# timeit: decorator to time functions
def timeit(f, single_time_data):
    def timed(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()

        """
        echo(
            style("[DEBUG] ", fg="green") + f"{f.__name__}  {((te - ts) * 1000):.2f} ms"
        )
        """

        single_time_data[f.__name__] = (te - ts) * 1000

        return result

    return timed


@njit(fastmath=True)
def cumsum(a: np.array) -> np.array:
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)

    return np.array(b)


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
    hist = np.zeros(256)

    # Get size of pixel array
    N = len(img_array)

    for l in range(256):
        for i in range(N):

            # Loop through pixels to calculate histogram
            if img_array.flat[i] == l:
                hist[l] += 1

    return hist


@njit(fastmath=True)
def calculate_histogram(img_array: np.array) -> (np.array, np.array, np.array):
    """
    g1(l) = ∑(l, k=0) pA(k) ⇒ g1(l)−g1(l −1) = pA(l) = hA(l)/NM (l = 1,...,255)

    geA(l) = round(255g1(l))

    calculate_histogram generates the histogram for an image,
    the equalized histogram,
    and a new quantized image based on the equalized histogram.
    """

    flat = img_array.flatten()

    hist = histogram(flat)
    cs = cumsum(hist)

    nj = (cs - cs.min()) * 255

    N = cs.max() - cs.min()

    cs = nj / N

    cs_casted = cs.astype(np.uint8)

    equalized = cs_casted[flat]
    img_new = np.reshape(equalized, img_array.shape)

    return hist, histogram(equalized), img_new


@njit(fastmath=True)
def mean_square_error(original_img: np.array, quantized_img: np.array) -> int:
    """
    mean_square_error takes two images represented as numpy arrays
    and finds the mean squared error of all of their pixel values.
    """

    mse = (np.square(original_img - quantized_img)).mean()

    return mse


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


@njit
def salt_pepper_noise(img_array: np.array, strength: int) -> np.array:
    """
    salt_pepper_noise randomly jumps through an image
    and converts a certain percentage of pixels white or black.
    The percentage is given in the strength parameter
    and the percentage of white to black pixels is 50%.
    """

    s_vs_p = 0.5
    out = np.copy(img_array)

    # Generate Salt '1' noise
    num_salt = np.ceil(strength * img_array.size * s_vs_p)

    for i in range(int(num_salt)):
        x = np.random.randint(0, img_array.shape[0] - 1)
        y = np.random.randint(0, img_array.shape[1] - 1)
        out[x][y] = 0

    # Generate Pepper '0' noise
    num_pepper = np.ceil(strength * img_array.size * (1.0 - s_vs_p))

    for i in range(int(num_pepper)):
        x = np.random.randint(0, img_array.shape[0] - 1)
        y = np.random.randint(0, img_array.shape[1] - 1)
        out[x][y] = 0

    return out


@njit
def gaussian_noise(img_array: np.array, sigma: int) -> np.array:
    """
    gaussian_noise creates a numpy array with the same dimensions as the image
    and generates gaussian normal noise based off the sigma parameter.
    That noise is added to a copy of the original image to achieve gaussian noise.
    """

    mean = 0.0

    noise = np.random.normal(mean, sigma, img_array.size)
    shaped_noise = noise.reshape(img_array.shape)

    gauss = img_array + shaped_noise

    return gauss


@njit(fastmath=True)
def apply_filter(img_array: np.array, img_filter: np.array) -> np.array:
    """
    Applies a linear filter to a copy of an image based on filter weights
    """

    rows, cols = img_array.shape
    height, width = img_filter.shape

    output = np.zeros((rows - height + 1, cols - width + 1))

    for rr in range(rows - height + 1):
        for cc in range(cols - width + 1):
            for hh in range(height):
                for ww in range(width):
                    imgval = img_array[rr + hh, cc + ww]
                    filterval = img_filter[hh, ww]
                    output[rr, cc] += imgval * filterval

    return output


def linear_filter(
    img_array: np.array, mask_size: int, weights: List[List[int]]
) -> np.array:

    """
    linear_filter uses a kernel or matrix of weights,
    given as a two dimensional List,
    and applies that kernel to a copy of an image.
    Applying the filter loops through every pixel in the image
    and multiples the values of the neighboring pixels by the weights in the kernel.
    The larger the kernel, the larger the neighborhood of pixels
    that affect the pixel being operated on at any given moment.
    """

    filter = np.array(weights)
    linear = apply_filter(img_array, filter)

    return linear


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


def export_image(img_arr: np.array, filename: str) -> None:
    """
    Exports a numpy array as a grey scale bmp image
    """
    img = Image.fromarray(img_arr)
    img = img.convert("L")
    img.save(conf["OUTPUT_DIR"] + filename + ".BMP")


def export_plot(img_arr: np.array, filename: str) -> None:
    """
    exports a historgam as a matplotlib plot
    """

    _ = plt.hist(img_arr, bins=256, range=(0, 256))
    plt.title(filename)
    plt.savefig(conf["OUTPUT_DIR"] + filename + ".png")
    plt.close()


def export_plots(plot_q: Queue()):
    """
    Exports a batch of histograms from a queue
    """

    plot_q.put(None)
    # 500 images * 2 plots per image
    ts = time.time()

    with tqdm(total=500 * 2) as pbar:
        for item in tqdm(iter(plot_q.get, None)):
            img_arr = item[0]
            filename = item[1]
            export_plot(img_arr, filename)
            pbar.update()

    te = time.time()

    echo(style("[INFO] ", fg="green") + f"exporting took {((te - ts) * 1000):.2f} ms")


def get_image_data(filename: Path) -> np.array:
    """
    Converts a bmp image to a numpy array
    """

    with Image.open(filename) as img:
        return np.array(img)


def apply_operations(img_file: Path, plot_q: Queue):
    """
    Runs an image through a set of operations
    """

    single_time_data = {}
    try:
        ts = time.time()

        color_img = get_image_data(img_file)

        # Grey scale image
        img = select_channel(color_img, color=conf["COLOR_CHANNEL"])

        # Create salt and peppered noise image
        salt_and_pepper = timeit(salt_pepper_noise, single_time_data)(
            img, conf["SALT_PEPPER_STRENGTH"]
        )

        # Create gaussian noise image
        gauss = timeit(gaussian_noise, single_time_data)(
            img, conf["GAUSS_NOISE_STRENGTH"]
        )

        # Apply linear filter to image
        linear = timeit(linear_filter, single_time_data)(
            img, conf["LINEAR_MASK"], conf["LINEAR_WEIGHTS"]
        )

        # Apply median filter to image
        median = timeit(median_filter, single_time_data)(
            img, conf["MEDIAN_MASK"], conf["MEDIAN_WEIGHTS"]
        )

        # Calculate histogram for image
        histogram, equalized, equalized_image = timeit(
            calculate_histogram, single_time_data
        )(img)

        msqe = mean_square_error(img, equalized_image)

        """
        echo(
            style(f"[DEBUG:{img_file.stem}] ", fg="green")
            + f"exporting plots and images for {img_file.stem}..."
        )
        """

        timeit(export_image, single_time_data)(
            salt_and_pepper, "salt_and_pepper_" + img_file.stem
        )

        timeit(export_image, single_time_data)(gauss, "gaussian_" + img_file.stem)

        timeit(export_image, single_time_data)(
            equalized_image, "equalized_" + img_file.stem
        )

        timeit(export_image, single_time_data)(linear, "linear_" + img_file.stem)

        timeit(export_image, single_time_data)(median, "median_" + img_file.stem)

        if platform == "darwin" or platform == "win32":
            plot_q.put((histogram, "histogram_" + img_file.stem))
            plot_q.put((equalized, "histogram_equalized_" + img_file.stem))
        else:
            timeit(export_plot, single_time_data)(
                histogram, "histogram_" + img_file.stem
            )
            timeit(export_plot, single_time_data)(
                equalized, "histogram_equalized_" + img_file.stem
            )

        te = time.time()
        single_time_data["total"] = (te - ts) * 1000

        return (
            style(f"{f'[INFO:{img_file.stem}]':15}", fg="green"),
            single_time_data,
            msqe,
        )

    except Exception as e:
        return (style(f"[ERROR:{img_file.stem}] ", fg="red") + str(e), {}, 0)


def parallel_operations(files: List[Path], plot_q: Queue):
    """
    Batch operates on a set of images in a multiprocess pool
    """

    time_data = Counter()

    image_operations = partial(apply_operations, plot_q=plot_q)

    echo(
        style("[INFO] ", fg="green")
        + f"initilizing process pool (number of processes: {conf['NUM_OF_PROCESSES']})"
    )
    echo(style("[INFO] ", fg="green") + "compiling...")
    with Pool(conf["NUM_OF_PROCESSES"]) as p:
        with tqdm(total=len(files)) as pbar:
            for res in tqdm(p.imap(image_operations, files)):
                pbar.write(res[0] + f" finished...   (msqe:{res[2]:8.2f})")
                pbar.update()

                time_data += res[1]

    return time_data


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
    global conf
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
    # DATA_SUBSET = 5
    # files = files[:DATA_SUBSET]

    with Manager() as manager:
        plot_q = manager.Queue()
        operation_time_data = parallel_operations(files, plot_q)
        if platform == "darwin" or platform == "win32":
            secho(
                "\n\n"
                + "[WARNING] MacOS and Windows require any GUI operation to be run in the main thread\n"
                + "          matplotlib uses Tkinter, a GUI library, to generate plots\n"
                + "          therefore plotting must be saved until the end and done synchronously\n\n"
                + "          Use linux or run inside a docker container for better performance\n\n",
                fg="red",
            )

            echo(style("[INFO] ", fg="green") + "exporting plots...")

            export_plots(plot_q)

    t_delta = time.time() - t0

    echo("\n\n")
    secho("Average operation time:", fg="green")
    for k, v in operation_time_data.items():
        echo(style("   ==> ", fg="green") + f"{k:20} : {(v / len(files)):8.2f} ms")

    print()
    secho(f"Total time: {t_delta:.2f} s", fg="green")


if __name__ == "__main__":
    main()
