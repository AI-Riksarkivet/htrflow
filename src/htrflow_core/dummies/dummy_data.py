import random

import cv2
import numpy as np


def create_blank_image(width: int, height: int, color=(255, 255, 255)):
    """Create new image(numpy array) filled with certain color in RGB"""
    image = np.full((height, width, 3), color, dtype=np.uint8)
    return image


def generate_handwriting_like_mask(
    image_width,
    image_height,
    segments=100,
    noise_level=1,
    bbox_top_left_factor=(0.1, 0.1),
    bbox_size_factor=(0.4, 0.1),
    smooth_level=5,
    seed=None,
):
    """
    Generates a mask that looks like an instance segmentation mask with irregular edges,
    allowing control over the bounding box's initial position and size, and applies smoothing.

    Parameters:
    - image_width, image_height: Dimensions of the image.
    - segments: Number of segments for each side of the bounding box to simulate polygon edges.
    - noise_factor: Factor of the bounding box's dimensions for added noise.
    - bbox_top_left_factor: A tuple (x_factor, y_factor) to determine the top left point of the bounding box
      as a factor of image dimensions.
    - bbox_size_factor: A tuple (width_factor, height_factor) to determine the size of the bounding box
      as a factor of image dimensions.
    - smooth_level: Factor of the image dimensions to determine the strength of the Gaussian blur.
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if smooth_level % 2 == 0:
        smooth_level += 1

    smooth_level = max(smooth_level, 1)

    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    bbox_top_left_x_factor, bbox_top_left_y_factor = bbox_top_left_factor
    bbox_width_factor, bbox_height_factor = bbox_size_factor

    # Calculate the top-left point and size of the bounding box
    top_left_x = int(image_width * bbox_top_left_x_factor)
    top_left_y = int(image_height * bbox_top_left_y_factor)
    bbox_width = int(image_width * bbox_width_factor)
    bbox_height = int(image_height * bbox_height_factor)

    # Adjust noise level based on bounding box size

    noise_level = min(noise_level / 100, 0.02)

    noise_level = max(bbox_width, bbox_height) * noise_level

    print(noise_level, noise_level)

    # TODO fix noise level sensitvity.
    # TODO fix so if its an bbox we skip rest of all the steps.. e.g noise-level is = 0?
    # TODO add so masks can be merged into complex shapes
    # TODO being able to control and create and create more complex shapes..
    #      perhaps use polygon to start with instead of bounding box?

    # Define the bounding box corners
    top_left = (top_left_x, top_left_y)

    bottom_right_x = top_left_x + bbox_width
    bottom_right_y = top_left_y + bbox_height
    bottom_right = (bottom_right_x, bottom_right_y)

    # Generate points along the rectangle edges
    x_points = np.linspace(top_left_x, bottom_right_x, num=segments // 4)
    y_points = np.linspace(top_left_y, bottom_right_y, num=segments // 4)

    # Generate the edges of the bounding box
    polygon_points = generate_polygon_edges(x_points, y_points, top_left, bottom_right, noise_level)

    # Convert points to the format expected by cv2.fillPoly and create the mask
    pts = np.array([polygon_points], np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], color=255)

    # Apply Gaussian blur for smoothing
    blur_strength = smooth_level  # Ensure odd size for kernel
    mask = cv2.GaussianBlur(mask, (blur_strength, blur_strength), 0)

    return mask


def generate_polygon_edges(x_points, y_points, top_left, bottom_right, noise_level):
    # Generates the noisy edges of the bounding box
    top_edge = [(x, top_left[1]) for x in x_points]
    bottom_edge = [(x, bottom_right[1]) for x in x_points]
    left_edge = [(top_left[0], y) for y in y_points[1:-1]]
    right_edge = [(bottom_right[0], y) for y in y_points[1:-1]]

    # Combine edges to form a closed loop
    polygon_points = top_edge + right_edge + bottom_edge[::-1] + left_edge[::-1]

    # Introduce noise to each point
    noisy_polygon_points = [
        (int(x + random.uniform(-noise_level, noise_level)), int(y + random.uniform(-noise_level, noise_level)))
        for x, y in polygon_points
    ]
    return noisy_polygon_points


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

    mask_1 = generate_handwriting_like_mask(
        width,
        height,
        segments=100,
        noise_level=1,
        bbox_top_left_factor=(0.2, 0.1),
        bbox_size_factor=(0.4, 0.12),
        smooth_level=3,
        seed=3,
    )

    mask_2 = generate_handwriting_like_mask(
        width,
        height,
        segments=100,
        noise_level=0.01,
        bbox_top_left_factor=(0.2, 0.15),
        bbox_size_factor=(0.3, 0.09),
        smooth_level=5,
        seed=3,
    )

    mask_3 = generate_handwriting_like_mask(
        width,
        height,
        segments=100,
        noise_level=1,
        bbox_top_left_factor=(0.5, 0.12),
        bbox_size_factor=(0.3, 0.54),
        smooth_level=5,
        seed=3,
    )

    final_masked_image = apply_mask_with_opacity(
        image,
        masks=[mask_1, mask_2],
        opacities=[0.5, 0.5],
        colors=[[255, 0, 0], [0, 255, 0]],
    )

    # Display the result
    cv2.imwrite("img.jpg", final_masked_image)
