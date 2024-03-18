"""
Module containing utilities related to images and geometries
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from htrflow_core.types.colors import Color, Colors
from htrflow_core.types.geometry import Bbox, Mask, Point, Polygon


if TYPE_CHECKING:
    from htrflow_core.results import Segment


# TODO is image thesame as Mask? Can we have an alias for that?


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
    labels: Optional[Sequence[str]] = None,
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
    return draw_polygons(image, polygons, color, thickness, alpha, labels)


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
    labels: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Draw polygons on image

    Args:
        image: The input image
        polygons: The polygons
        color: Fill and border color
        alpha: Opacity of the fill
        labels: Polygon labels

    Returns:
        A copy of the input image with polygons drawn on it.
    """
    image = image.copy()
    cv2.polylines(image, polygons, isClosed=True, color=color, thickness=thickness)
    if alpha > 0:
        for polygon in polygons:
            filled = cv2.fillPoly(image.copy(), [polygon], color=color)
            image = image * (1 - alpha) + filled * alpha

    labels = labels if labels else []
    for label, polygon in zip(labels, polygons):
        x = min(x for x, _ in polygon)
        y = min(y for _, y in polygon)
        image = draw_label(image, label, (x, y), bg_color=color, font_thickness=thickness)

    return image


def draw_label(
    image: np.ndarray,
    label: str,
    pos: tuple[int, int],
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: int = 2,
    font_thickness: int = 2,
    font_color: Color = Colors.WHITE,
    bg_color: Optional[Color] = Colors.BLACK,
):
    """Draw label on image

    This function draws text on the image. It tries to put the text at
    the given position, but will move it downwards if the requested
    position would put the text above the image.

    Arguments:
        image: Input image
        label: Text to write on image
        pos: (x, y) coordinate of the text's bottom left corner
        font: A cv2 font
        font_scale: Font scale
        font_thickness: Font thickness
        font_color: Font color
        bg_color: Optional background color
    """

    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Adjust y coordinate if needed. It must be at least equal to the
    # height of the text to ensure that the text is visible, or else
    # the text ends up above the image.
    pos = pos[0], max(pos[1], text_h)

    if bg_color is not None:
        x, y = pos
        pad = font_thickness
        p1 = x - pad, y + pad
        p2 = x + text_w + pad, y - text_h + pad
        image = cv2.rectangle(image, p1, p2, bg_color, -1)

    return cv2.putText(image, label, pos, font, font_scale, font_color, font_thickness)


def draw_reading_order():
    pass


def mask2polygon(mask: Mask, epsilon: float = 0.005) -> Polygon:
    """Convert mask to polygon

    Args:
        mask: The input mask
        epsilon: A tolerance parameter. Smaller epsilon will result in
            a higher-resolution polygon.

    Returns:
        A list of coordinate tuples
    """
    # Ensure mask is 8-bit single-channel image
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if len(mask.shape) == 3 and mask.shape[2] != 1:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

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
    return cv2.imread(source)


def write(dest: str, image: np.ndarray) -> None:
    cv2.imwrite(dest, image)


def helper_plot_for_segment(
    image: np.ndarray,
    segment_results: List[Segment],
    maskcolor: Optional[Color] = Colors.RED,
    maskalpha: float = 0.4,
    boxcolor: Optional[str] = "blue",
    polygencolor: Optional[str] = "yellow",
    fontcolor: Optional[str] = "white",
    fontsize: Optional[int] = None,
) -> None:
    """
    Displays an image with mask overlays, bounding boxes, polygons, class labels, and scores using Matplotlib.
    The function provides optional parameters to customize the appearance of these elements, including their colors and the mask's opacity.
    The font size for text annotations can be specified; if not provided, it will be dynamically adjusted based on the average size of the bounding boxes.

    Args:
        image: Background image as a NumPy array.
        segment_results: List of Segment objects for overlay. Each segment contains the bounding box, mask, class label, score, and polygon information.
        maskcolor: Optional; fill color for masks. The default is RED in RGB format. Set to None to disable mask overlays.
        maskalpha: Float specifying the opacity level of mask overlays. Default is 0.4.
        boxcolor: Optional; color for bounding box edges. The default is BLUE. Set to None to disable drawing bounding boxes.
        polygencolor: Optional; color for polygon edges. The default is YELLOW. Set to None to disable drawing polygons.
        fontcolor: Optional; color for the text annotations (class labels and scores). The default is WHITE.
        fontsize: Optional; integer specifying the font size for text annotations. If not provided, the size will be dynamically determined based on the average bounding box size.
    """

    maskcolor = bgr_to_rgb(maskcolor)

    if fontsize is not None:
        avg_bbox_size = sum(
            (x2 - x1 + y2 - y1) / 2 for x1, x2, y1, y2 in (segment.bbox for segment in segment_results)
        ) / len(segment_results)
        fontsize = max(24, min(8, avg_bbox_size / 100))

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for index, segment in enumerate(segment_results):
        bbox, mask, score, class_label, polygon = (
            segment.bbox,
            segment.mask,
            segment.score,
            segment.class_label,
            segment.polygon,
        )

        x1, x2, y1, y2 = bbox

        if maskcolor is not None:
            rgba_mask = mask_to_rgba(mask, (x1, y1), image.shape[:2], maskcolor, maskalpha)
            ax.imshow(rgba_mask, interpolation="none")

        if boxcolor is not None:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=boxcolor, facecolor="none")
            ax.add_patch(rect)

        if polygencolor is not None:
            poly_patch = patches.Polygon(polygon, linewidth=1, edgecolor=polygencolor, facecolor="none")
            ax.add_patch(poly_patch)

        if fontcolor is not None:
            label_text = f"{index}: Class: {class_label}, Score: {score:.2f}"
            ax.text(
                x1,
                y1,
                label_text,
                color=fontcolor,
                fontsize=fontsize,
                bbox={"facecolor": "black", "alpha": 0.4, "pad": 0, "edgecolor": "none"},
            )

    plt.show()


def mask_to_rgba(mask: Mask, point: Point, image_shape: Tuple[int, int], maskcolor: Color, alpha: float) -> np.ndarray:
    """
    Creates an RGBA overlay from a binary mask with specified color and alpha.

    Args:
        mask: Binary mask array.
        point: Top-left coordinate (x1, y1) for mask positioning.
        image_shape: Shape (height, width) of the background image.
        maskcolor: Color for the mask overlay.
        alpha: Transparency factor (0 is transparent, 1 is opaque).

    Returns:
        An RGBA image for overlaying on the background image.
    """
    x1, y1 = point
    rgba_mask = np.zeros((image_shape[0], image_shape[1], 4), dtype=np.uint8)

    nonzero_y, nonzero_x = np.nonzero(mask)
    adjusted_nonzero_x = nonzero_x + x1
    adjusted_nonzero_y = nonzero_y + y1
    rgba_mask[adjusted_nonzero_y, adjusted_nonzero_x] = [*maskcolor, int(255 * alpha)]

    return rgba_mask


def bgr_to_rgb(bgr_color: Color) -> Color:
    """Convert a color from BGR to RGB."""
    return tuple(reversed(bgr_color))
