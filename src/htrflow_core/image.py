"""
Module containing utilities related to images and geometries
"""

from typing import Iterable

import cv2
import numpy as np


Color = tuple[int, int, int]
Mask = np.ndarray[np.uint8]
Polygon = Iterable[tuple[int, int]]
Bbox = tuple[int, int, int, int]


class Colors:
    """Color constants in BGR"""

    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)


def crop(image: np.ndarray, bbox: Bbox) -> np.ndarray:
    """Crop image

    Args:
        image: The input image
        bbox: The bounding box
    """
    x1, x2, y1, y2 = bbox
    return image[y1:y2, x1:x2]


def mask(
    image: np.ndarray,
    mask: Mask,
    fill: Color = Colors.WHITE,
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


def draw_bboxes(
    image: np.ndarray,
    bboxes: Iterable[Bbox],
    color: Color = Colors.BLUE,
    thickness: int = 3,
    alpha: float = 0.2,
) -> np.ndarray:
    """Draw bounding boxes on image

    Args:
        image: The input image
        bboxes: List of bounding boxes to draw
        color: Box border color
        thickness: Box border thickness
    Returns:
        A copy of the input image with the bounding boxes drawn.
    """
    polygons = [bbox2polygon(bbox) for bbox in bboxes]
    return draw_polygons(image, polygons, color, thickness, alpha)


def draw_masks(
    image: np.ndarray,
    masks: Iterable[Mask],
    color: Color = Colors.BLUE,
    alpha: float = 0.2,
) -> np.ndarray:
    """Draw masks on image

    Args:
        image: The input image
        masks: The masks
        color: Mask color
        alpha: Mask opacity
    Returns:
        A copy of the input image with the masked areas colored.
    """
    for mask_ in masks:
        masked = mask(image, mask_, inverse=True, fill=color)
        image = image * (1 - alpha) + masked * alpha
    return image


def draw_polygons(
    image: np.ndarray,
    polygons: Iterable[Polygon],
    color: Color = Colors.BLUE,
    thickness: int = 3,
    alpha: float = 0.2,
) -> np.ndarray:
    """Draw polygons on image

    Args:
        image: The input image
        polygons: The polygons
        color: Fill and border color
        alpha: Opacity of the fill

    Returns:
        A copy of the input image with polygons drawn on it.
    """
    image = image.copy()
    cv2.polylines(image, polygons, isClosed=True, color=color, thickness=thickness)
    if alpha > 0:
        for polygon in polygons:
            filled = cv2.fillPoly(image.copy(), [polygon], color=color)
            image = image * (1 - alpha) + filled * alpha
    return image


def mask2polygon(mask: Mask, epsilon: float = 0.01) -> Polygon:
    """Convert mask to polygon

    Args:
        mask: The input mask
        epsilon: A tolerance parameter. Smaller epsilon will result in
            a higher-resolution polygon.

    Returns:
        A list of coordinate tuples
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # `contours` is a high-resolution contour, but we need a simple polygon, so
    # we approximate it with the DP algorithm, which removes as many points as
    # possible while still keeping the original shape.

    # Adjust the tolerance parameter `epsilon` relative to the size of the mask
    epsilon *= cv2.arcLength(contours[0], closed=True)
    approx = cv2.approxPolyDP(contours[0], epsilon, closed=True)
    return np.squeeze(approx)


def masks2polygons(masks: Iterable[Mask], epsilon=0.01) -> Iterable[Polygon]:
    """Convert masks to polygons"""
    return [mask2polygon(mask, epsilon) for mask in masks]


def bbox2polygon(bbox: Bbox) -> Polygon:
    """Convert bounding box to polygon"""
    x1, x2, y1, y2 = bbox
    return np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])


def mask2bbox(mask: Mask) -> Bbox:
    """Convert mask to bounding box"""
    y, x = np.where(mask != 0)
    return np.min(x), np.max(x), np.min(y), np.max(y)
