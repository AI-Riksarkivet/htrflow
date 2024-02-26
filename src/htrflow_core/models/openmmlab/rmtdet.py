# from htrflow_core.helper.timing_decorator import timing_decorator
# from htrflow_core.inferencers.base_inferencer import BaseInferencer
# from htrflow_core.models.openmmlab_loader import OpenmmlabModel


# # from htrflow.structures.result import Result, SegResult


# class MMDetInferencer(BaseInferencer):
#     def __init__(self, region_model: OpenmmlabModel, parent_result=None):
#         self.region_model = region_model.model
#         self.parent_result = parent_result

#     def preprocess():
#         pass

#     @timing_decorator
#     def predict(self, input_images, batch_size: int = 1, pred_score_thr: float = 0.3):
#         # image = mmcv.imread(input_images)

#         result_raw = self.region_model(input_images, batch_size, return_datasample=True)
#         batch_result = self.postprocess(result_raw, input_images)

#         return batch_result

#     @timing_decorator
#     def postprocess(self, result_raw, input_images):
#         pass

#     #     # hm jag behöver nog inte nästla lines i seg_result, eller kanske ett bra sätt för att skilja på lines within regions och lines within page?
#     #     # i postprocess... eller hmm, jag vill inte lägga för mycket här, jag tror att jag lägger det i en egen klass istället...

#     #     batch_result = [
#     #         Result(
#     #             img_shape=np.shape(y)[0:2],
#     #             segmentation=SegResult(
#     #                 labels=x.pred_instances.labels.clone(),
#     #                 bboxes=x.pred_instances.bboxes.clone(),
#     #                 masks=x.pred_instances.masks.clone(),
#     #                 scores=x.pred_instances.scores.clone(),
#     #             ),
#     #         )
#     #         for x, y in zip(result_raw["predictions"], input_images)
#     #     ]

#     #     return batch_result
#     def align_masks_with_image(self, img):
#         # Create a tensor of all masks
#         all_masks = self.masks

#         # Calculate the size of the image
#         img_size = (img.shape[0], img.shape[1])

#         # Resize and pad each mask to match the size of the image
#         masks = []
#         for i in range(all_masks.shape[0]):
#             mask = all_masks[i]

#             # Convert the mask to float
#             mask_float = mask.float()

#             # Resize the mask
#             mask_resized = torch.nn.functional.interpolate(mask_float[None, None, ...], size=img_size, mode="nearest")[
#                 0, 0
#             ]

#             # Convert the mask back to bool
#             mask = mask_resized.bool()

#             # Pad the mask
#             padded_mask = torch.zeros(img_size, dtype=torch.bool, device=mask.device)
#             padded_mask[: mask.shape[0], : mask.shape[1]] = mask
#             mask = padded_mask

#             masks.append(mask)

#         # Stack all masks into a single tensor
#         self.masks = torch.stack(masks)
