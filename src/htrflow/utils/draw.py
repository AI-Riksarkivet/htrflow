"""
Visualization utilities
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Tuple, TypeAlias

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from htrflow.utils import imgproc
from htrflow.utils.geometry import Bbox, Mask, Point, Polygon


if TYPE_CHECKING:
    from htrflow.results import Segment


Color: TypeAlias = Tuple[int, int, int]


class Colors:
    """Color constants in BGR."""

    RED: Color = (0, 0, 255)
    GREEN: Color = (0, 255, 0)
    BLUE: Color = (255, 0, 0)
    WHITE: Color = (255, 255, 255)
    BLACK: Color = (0, 0, 0)


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
    polygons = [bbox.polygon() for bbox in bboxes]
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
        masked = imgproc.mask(image, mask_, inverse=True, fill=color)
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
    polygons = [polygon.as_nparray() for polygon in polygons]
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
    """draw_reading_order()

    Not yet implemented
    """
    pass


def helper_plot_for_segment(
    segment_results: List[Segment],
    image: Optional[np.ndarray] = None,
    maskcolor: Optional[Color] = Colors.RED,
    maskalpha: float = 0.4,
    boxcolor: Optional[str] = "blue",
    polygencolor: Optional[str] = "yellow",
    fontcolor: Optional[str] = "white",
    fontsize: Optional[int] = None,
    figsize: Optional[tuple] = (10, 8),
    save_fig: Optional[str] = "foo.png",
) -> None:
    """
    Displays an image with mask overlays, bounding boxes, polygons,
    class labels, and scores using Matplotlib. The function provides
    optional parameters to customize the appearance of these
    elements, including their colors and the mask's opacity.
    The font size for text annotations can be specified; if not
    provided, it will be dynamically adjusted based on the average
    size of the bounding boxes.

    Args:
        segment_results: List of Segment objects for overlay. Each
            segment contains the bounding box, mask, class label,
            score, and polygon information.
        image: Background image as a NumPy array. If None, will try to use
            background images as an empty mask from Segments.orig_shape
        maskcolor: Optional; fill color for masks. The default is RED
            in RGB format. Set to None to disable mask overlays.
        maskalpha: Float specifying the opacity level of mask
            overlays. Default is 0.4.
        boxcolor: Optional; color for bounding box edges. The default
            is BLUE. Set to None to disable drawing bounding boxes.
        polygencolor: Optional; color for polygon edges. The default
            is YELLOW. Set to None to disable drawing polygons.
        fontcolor: Optional; color for the text annotations (class
            labels and scores). The default is WHITE.
        fontsize: Optional; integer specifying the font size for text
            annotations. If not provided, the size will be dynamically
            determined based on the average bounding box size.
        figsize: Optional; tuple specifying the figure size in inches.
            The default is (10, 8).
        save_fig: Save image locally
    """

    if image is None:
        img_shape = segment_results[0].orig_shape
        image = np.zeros(img_shape, dtype=np.uint8)

    if fontsize is not None:
        avg_bbox_size = sum(
            (x2 - x1 + y2 - y1) / 2 for x1, y1, x2, y2 in (segment.bbox for segment in segment_results)
        ) / len(segment_results)
        fontsize = max(24, min(8, avg_bbox_size / 100))

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    for index, segment in enumerate(segment_results):
        bbox, mask, score, class_label, polygon = (
            segment.bbox,
            segment.mask,
            segment.score,
            segment.class_label,
            segment.polygon,
        )

        x1, y1, x2, y2 = bbox

        if maskcolor is not None:
            maskcolor_rgba = bgr_to_rgb(maskcolor)
            if mask is not None:
                rgba_mask = mask_to_rgba(mask, (x1, y1), image.shape[:2], maskcolor_rgba, maskalpha)
                ax.imshow(rgba_mask, interpolation="none")

        if boxcolor is not None:
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1,
                edgecolor=boxcolor,
                facecolor="none",
            )
            ax.add_patch(rect)

        if polygencolor is not None:
            poly_patch = patches.Polygon(
                polygon.as_nparray(),
                linewidth=1,
                edgecolor=polygencolor,
                facecolor="none",
            )
            ax.add_patch(poly_patch)

        if fontcolor is not None:
            class_label_str = class_label if class_label is not None else "Unknown"
            score_str = f"{score:.2f}" if score is not None else "N/A"
            reading_order_str = f"RO: {index}, " if len(segment_results) > 1 else ""

            label_text = f"{reading_order_str}Class: {class_label_str}, Score: {score_str}"
            ax.text(
                x1,
                y1,
                label_text,
                color=fontcolor,
                fontsize=fontsize,
                bbox={
                    "facecolor": "black",
                    "alpha": 0.4,
                    "pad": 0,
                    "edgecolor": "none",
                },
            )

    if save_fig:
        import os

        cache_dir = ".cache"
        os.makedirs(cache_dir, exist_ok=True)
        plt.savefig(os.path.join(cache_dir, save_fig))
    else:
        plt.show()


def mask_to_rgba(
    mask: Mask,
    point: Point,
    image_shape: Tuple[int, int],
    maskcolor: Color,
    alpha: float,
) -> np.ndarray:
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
