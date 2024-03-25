"""
Image processing utilities
"""

import cv2
import numpy as np

from htrflow_core.utils.geometry import Bbox, Mask


def crop(image: np.ndarray, bbox: Bbox) -> np.ndarray:
    """Crop image

    Args:
        image: The input image
        bbox: The bounding box
    """
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


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
    # Moved from binarize.py
    # TODO: Double check color space conversions (other functions in this module operate on BGR)
    img_ori = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    dst = cv2.fastNlMeansDenoising(img_gray, h=31, templateWindowSize=7, searchWindowSize=21)
    img_blur = cv2.medianBlur(dst, 3).astype("uint8")
    threshold = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img_binarized = cv2.cvtColor(threshold, cv2.COLOR_BGR2RGB)
    return img_binarized


def read(source: str) -> np.ndarray:
    img = cv2.imread(source)
    if img is None:
        raise RuntimeError(f"Could not load {source}")
    return img


def write(dest: str, image: np.ndarray) -> None:
    cv2.imwrite(dest, image)
