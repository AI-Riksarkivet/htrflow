"""
Image processing utilities
"""

import logging
import re
from typing import Any, TypeAlias

import cv2
import numpy as np
import numpy.typing as npt
import requests

from htrflow.utils.geometry import Bbox, Mask


NumpyImage: TypeAlias = np.ndarray  # TODO make non-generic
logger = logging.getLogger(__name__)


def crop(image: npt.NDArray[Any], bbox: Bbox, padding: int | None = 0) -> npt.NDArray[Any]:
    """Crop image

    Args:
        image: The input image
        bbox: The bounding box
        padding: A value to pad the cropped image with if the given bounding
            box overflows the input image. This ensures that the cropped image
            has the same size as the bounding box. If None, no padding is used,
            and the shape of the cropped image may not match the bounding box.
    """
    x1, y1, x2, y2 = bbox
    cropped = image[y1:y2, x1:x2].copy()
    h, w = cropped.shape[:2]
    if padding is not None and (h < bbox.height or w < bbox.width):
        pad_y = bbox.height - h
        pad_x = bbox.width - w
        cropped = np.pad(cropped, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=padding)
    return cropped


def mask(
    image: npt.NDArray[Any],
    mask: Mask,
    fill: tuple[int, int, int] = (255, 255, 255),
    inverse: bool = False,
) -> npt.NDArray[Any]:
    """Apply mask to image

    Returns a copy of the input image where all pixels such that mask[pixel]
    is False are filled with the specified color. Uses imgproc.crop to crop
    (or possibly pad) the mask if necessary.

    Args:
        image: The input image
        mask: The mask, a binary array, of similar shape as `image`
        fill: The color value to fill the masked areas with, default white
        inverse: Fill all pixels where mask[pixel] is True instead of False
    """
    image = image.copy()
    idx = (mask != 0) if inverse else (mask == 0)
    if idx.shape != image.shape[:2]:
        logger.debug(
            "Resizing mask to match the input image (mask %d-by-%d, image %d-by-%d)",
            *idx.shape,
            *image.shape[:2],
        )
        idx = crop(idx, Bbox(0, 0, image.shape[1], image.shape[0]))
    image[idx] = fill
    return image


def resize(image: npt.NDArray[Any], shape: tuple[int, int]) -> npt.NDArray[Any]:
    """Resize image using nearest-neighbour interpolation

    Arguments:
        image: Input image
        shape: Desired shape as a (height, width) tuple
    """
    if shape == image.shape[:2]:
        return image
    y, x = shape
    return cv2.resize(image, (x, y), interpolation=cv2.INTER_NEAREST)


def rescale(image: npt.NDArray[Any], ratio: float) -> npt.NDArray[Any]:
    """Rescale image

    Rescales the image while keeping the aspect ratio as far as
    possible. Uses nearest-neighbour interpolation.

    Arguments:
        image: Input image
        ratio: Ratio of size of rescaled image to its original size in
            pixels, i.e.
                ratio = (pixels in rescaled) / (pixels in original)
            For example, with ratio=0.25 a 200x100 image would be resized
            to 100x50.
    """
    length_ratio = np.sqrt(ratio)
    return rescale_linear(image, length_ratio)


def rescale_linear(image: npt.NDArray[Any], ratio: float) -> npt.NDArray[Any]:
    """Rescale image

    Rescales the image while keeping the aspect ratio intact as far as
    possible. Uses nearest-neighbour interpolation.

    The only difference between this function and imgproc.rescale is that
    this function applies the rescaling factor on the side lengths and not
    on the total area. This function thus produces smaller images than
    imgproc.rescale with the same scaling factor.

    Arguments:
        image: Input image
        ratio: Ratio of rescaled side length to original side length, i.e.
                ratio = (side length rescaled) / (side length originl)
            For example, with ratio=0.25 a 200x100 image would be resized to
            50x25.
    """
    h, w = image.shape[:2]
    return resize(image, (int(h * ratio), int(w * ratio)))


def binarize(image: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Binarize image"""
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.fastNlMeansDenoising(img_gray, h=31, templateWindowSize=7, searchWindowSize=21)
    img_gray = cv2.medianBlur(dst, 3).astype("uint8")
    threshold = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)


def is_http_url(string: str) -> bool:
    """Check if the string is a valid HTTP URL."""
    return re.match(r"^https?://", string, re.IGNORECASE) is not None


def _is_valid_url(url: str) -> bool:
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def read(source: str | npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Read an image from a URL, a local path, or directly use a numpy array as an OpenCV image.

    Args:
        source: The source can be a URL, a local filesystem path, or a numpy array representing an image.

    Returns:
        np.ndarray: Image in OpenCV format.

    Raises:
        ImageImportError: If the image cannot be loaded from the given source.
        TypeError: It the type of `source` is not string or numpy array.
    """
    if not isinstance(source, str):
        raise TypeError(f"Type of `source` should be string or numpy image, not {type(source)}")

    # Return the image as-is if it already is a numpy array
    if isinstance(source, np.ndarray):
        return source

    error_msg = f"Could not load an image from {source}. "

    # Try to load from URL
    if is_http_url(source):
        if not _is_valid_url(source):
            raise ImageImportError(error_msg + "The URL is invalid or unreachable.")
        resp = requests.get(source, stream=True).raw
        image_arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        img = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ImageImportError(error_msg + "The URL could not be interpreted as an image.")
        return img

    # Try to load from filesystem
    img = cv2.imread(source, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageImportError(error_msg + "Check that the path exists and is a valid image.")
    return img


def write(dest: str, image: npt.NDArray[Any]) -> str:
    cv2.imwrite(dest, image)
    logger.info("Wrote image to %s", dest)
    return dest


class ImageImportError(RuntimeError):
    pass
