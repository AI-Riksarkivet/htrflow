"""
Image processing utilities
"""

import os
import re

import cv2
import numpy as np
import requests

from htrflow_core.utils.geometry import Bbox, Mask


def crop(image: np.ndarray, bbox: Bbox) -> np.ndarray:
    """Crop image

    Args:
        image: The input image
        bbox: The bounding box
    """
    x1, y1, x2, y2 = bbox
    return image[y1 : y2 + 1, x1 : x2 + 1]


def mask(
    image: np.ndarray,
    mask: Mask,
    fill: tuple[int, int, int] = (255, 255, 255),
    inverse: bool = False,
) -> np.ndarray:
    """Apply mask to image

    Args:
        image: The input image
        mask: The mask, a binary array, of the same shape as `image`
        fill: The color value to fill the masked areas with, default white
        inverse: Invert the mask before applying
    """
    image = image.copy()
    idx = (mask != 0) if inverse else (mask == 0)
    image[idx] = fill
    return image


def binarize(image: np.ndarray) -> np.ndarray:
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


def read(source: str | np.ndarray | os.PathLike) -> np.ndarray:
    """Read an image from a URL, a local path, or directly use a numpy array as an OpenCV image.

    Args:
        source: The source can be a URL, a local filesystem path, or a numpy array representing an image.

    Returns:
        np.ndarray: Image in OpenCV format.

    Raises:
        RuntimeError: If the image cannot be loaded from the given source.
        ValueError: If the source type is unsupported.
    """
    if isinstance(source, np.ndarray):
        return source
    elif isinstance(source, str):
        if is_http_url(source):
            if not _is_valid_url(source):
                raise ValueError("The URL is invalid or unreachable.")
            resp = requests.get(source, stream=True).raw
            image_arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            img = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Could not load the image from {source}")
            return img
        else:
            img = cv2.imread(source, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Could not load the image from {source}")
            return img
    else:
        raise ValueError("Source must be a string URL, np.ndarray, or a filesystem path")


def write(dest: str, image: np.ndarray) -> None:
    cv2.imwrite(dest, image)
