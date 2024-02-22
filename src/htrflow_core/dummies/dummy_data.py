import random

import cv2
import numpy as np


def create_blank_image(width: int, height: int, color=(255, 255, 255)):
    """Create new image(numpy array) filled with certain color in RGB"""
    image = np.full((height, width, 3), color, dtype=np.uint8)
    return image


def generate_handwriting_like_mask(
    image_width=210 * 2,
    image_height=297,
    text_height_factor=0.1,
    variation_factor=0.01,
    horizontal_start_position_factor=0.05,
    mask_length_factor=0.4,
    vertical_start_position_factor=0.1,
    size_reduction_factor=0,
):
    mask = np.zeros((image_height, image_width), dtype=np.uint8)  # Single channel mask

    # Calculate dimensions and positions based on factors
    text_height = int(image_height * text_height_factor)
    variation = int(image_height * variation_factor)
    horizontal_start = int(image_width * horizontal_start_position_factor)
    mask_length = int(image_width * mask_length_factor)
    size_reduction = int(text_height * size_reduction_factor)

    baseline_y = int(image_height * vertical_start_position_factor)
    adjusted_text_height = text_height - size_reduction

    start_x = horizontal_start
    end_x = start_x + mask_length
    end_x = min(end_x, image_width)  # Ensure the mask does not exceed image width

    top_boundary = []
    bottom_boundary = []

    # Generate boundary points
    x_points = np.linspace(start_x, end_x, num=(end_x - start_x) // 40)

    for x in x_points:
        top_y_variation = random.randint(-variation, variation)
        bottom_y_variation = random.randint(-variation, variation)

        top_y = baseline_y - adjusted_text_height // 2 + top_y_variation
        bottom_y = baseline_y + adjusted_text_height // 2 + bottom_y_variation

        top_boundary.append((x, top_y))
        bottom_boundary.append((x, bottom_y))

    # Ensure smooth closure of the mask
    polyline_points = np.array(top_boundary + bottom_boundary[::-1] + [top_boundary[0]], np.int32)
    cv2.fillPoly(mask, [polyline_points], color=255)

    return mask


def apply_mask_with_opacity(image, masks, opacities, colors):
    for mask, opacity, color in zip(masks, opacities, colors):
        color_overlay = np.zeros_like(image)
        color_overlay[:, :] = color

        binary_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        binary_mask[mask == 255] = [255, 255, 255]
        binary_mask[mask == 0] = [0, 0, 0]

        image = np.where(
            binary_mask == [255, 255, 255], cv2.addWeighted(image, 1 - opacity, color_overlay, opacity, 0), image
        )

    return image


if __name__ == "__main__":
    width, height = 210 * 2, 297

    white = (255, 255, 255)
    image = create_blank_image(width, height, white)

    # Original mask
    first_mask = generate_handwriting_like_mask(
        width,
        height,
        text_height_factor=0.2,
        variation_factor=0.01,
        horizontal_start_position_factor=0.05,
        mask_length_factor=0.4,
        vertical_start_position_factor=0.15,
    )

    secondary_mask = generate_handwriting_like_mask(
        width,
        height,
        text_height_factor=0.18,
        variation_factor=0.01,
        horizontal_start_position_factor=0.1,
        mask_length_factor=0.34,
        vertical_start_position_factor=0.16,
        size_reduction_factor=0.02,
    )

    third_mask = generate_handwriting_like_mask(
        width,
        height,
        text_height_factor=0.2,
        variation_factor=0.01,
        horizontal_start_position_factor=0.05,
        mask_length_factor=0.4,
        vertical_start_position_factor=0.7,
        size_reduction_factor=0.02,
    )

    forth_mask = generate_handwriting_like_mask(
        width,
        height,
        text_height_factor=0.3,
        variation_factor=0.01,
        horizontal_start_position_factor=0.55,
        mask_length_factor=0.4,
        vertical_start_position_factor=0.2,
        size_reduction_factor=0.02,
    )

    final_masked_image = apply_mask_with_opacity(
        image,
        masks=[first_mask, secondary_mask, third_mask, forth_mask],
        opacities=[0.5, 0.5, 0.5, 0.5],
        colors=[[255, 0, 0], [0, 255, 0], [20, 100, 20], [0, 0, 255]],
    )

    # Display the result
    cv2.imwrite("img.jpg", final_masked_image)
