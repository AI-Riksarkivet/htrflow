"""
Image processing utilities
"""

import logging

import cv2
import numpy as np
from PIL import Image

from htrflow.utils.geometry import Polygon, polygon2mask


logger = logging.getLogger(__name__)


def mask(image: Image, mask: np.ndarray, fill: tuple[int, int, int] = (255, 255, 255)) -> Image:
    """Apply mask to image

    Returns a copy of the input image where all pixels such that mask[pixel]
    is False are filled with the specified color. Uses imgproc.crop to crop
    (or possibly pad) the mask if necessary.

    Args:
        image: The input image
        mask:
        fill: The color value to fill the masked areas with, default white
    """

    mask = Image.fromarray(mask, mode="L")
    if image.size != mask.size:
        print(image.size, mask.size)
        logger.debug("Resizing mask to match the input image")
        mask = mask.crop((0, 0, *image.size))

    fill = Image.new(image.mode, image.size, color=fill)
    return Image.composite(image, fill, mask)


def polygon_mask(image: Image, polygon: Polygon):
    """
    Apply polygon mask to image

    Arguments:
        image: Input image
        polygon: Polygon to use as mask

    Returns:
        A copy of `image` with everything outside `polygon` is masked.
    """
    bbox = polygon.bbox
    image = image.crop(bbox)
    polygon = polygon.move(-bbox.p1)
    return mask(image, polygon2mask(polygon, (bbox.height, bbox.width)))


def rescale(image: Image, ratio: float) -> Image:
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


def rescale_linear(image: Image, ratio: float) -> Image:
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
    return Image.resize((int(image.width * ratio), int(image.height * ratio)))


def binarize(image: Image) -> Image:
    """Binarize image"""
    image = np.asarray(image)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.fastNlMeansDenoising(img_gray, h=31, templateWindowSize=7, searchWindowSize=21)
    img_gray = cv2.medianBlur(dst, 3).astype("uint8")
    threshold = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR))


class ImageImportError(RuntimeError):
    pass
