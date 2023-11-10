import cv2
import numpy as np
import torch

from htrflow.structures.result import Result
from htrflow.utils.helper import timing_decorator


class PostProcessSegmentation:
    def __init__(self):
        pass

    def get_bounding_box(mask):
        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        xmin, xmax = torch.where(cols)[0][[0, -1]]

        return xmin, ymin, xmax, ymax

    def get_bounding_box_np(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        return xmin, ymin, xmax, ymax

    @staticmethod
    @timing_decorator
    def crop_imgs_from_result_optim(result: Result, img):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Convert img to a PyTorch tensor and move to GPU if available
        img = torch.from_numpy(img).to(device)

        cropped_imgs = []
        masks = result.segmentation.masks.to(device)

        for mask in masks:
            # Get bounding box
            xmin, ymin, xmax, ymax = PostProcessSegmentation.get_bounding_box(mask)

            # Crop masked region and put on white background
            masked_region = img[ymin : ymax + 1, xmin : xmax + 1]
            white_background = torch.ones_like(masked_region) * 255

            # Apply mask to the image
            masked_region_on_white = torch.where(
                mask[ymin : ymax + 1, xmin : xmax + 1][..., None], masked_region, white_background
            )
            masked_region_on_white_np = masked_region_on_white.cpu().numpy()

            cropped_imgs.append(masked_region_on_white_np)

        return cropped_imgs
    
    def combine_region_line_res(result_full, result_regions):
        ind = 0

        for res in result_full: 
            res.nested_results = list()   
            for i in range(ind, ind + len(res.segmentation.masks)):
                #result_lines.parent_result = res
                res.nested_results.append(result_regions[i])
            
            ind += len(res.segmentation.masks)
            
        