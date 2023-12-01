from htrflow.helper.timing_decorator import timing_decorator
from htrflow.models.openmmlab_loader import OpenmmlabModelLoader
from htrflow.postprocess.postprocess_segmentation import PostProcessSegmentation
from htrflow.postprocess.postprocess_transcription import PostProcessTranscription


def post_process_seg(result, imgs, lines=False, regions=False):
    imgs_cropped = []

    for res, img in zip(result, imgs):
        res.segmentation.remove_overlapping_masks()
        res.segmentation.align_masks_with_image(img)

        if regions:
            res.order_regions_marginalia(img)
        elif lines:
            res.order_lines()

        imgs_cropped.append(PostProcessSegmentation.crop_imgs_from_result_optim(res, img))

    return result, imgs_cropped


@timing_decorator
def predict_batch(inferencer_regions, inferencer_lines, inferencer_htr, imgs_numpy):
    result_full = inferencer_regions.predict(imgs_numpy, batch_size=8)

    imgs_region_numpy = []

    result_full, imgs_region_numpy = post_process_seg(result_full, imgs_numpy, regions=True)
    flat_imgs_region_numpy = [item for sublist in imgs_region_numpy for item in sublist]
    result_regions = inferencer_lines.predict(flat_imgs_region_numpy, batch_size=8)
    result_regions, imgs_lines_numpy = post_process_seg(result_regions, flat_imgs_region_numpy, lines=True)

    PostProcessSegmentation.combine_region_line_res(result_full, result_regions)

    flat_imgs_lines_numpy = [item for sublist in imgs_lines_numpy for item in sublist]

    print(len(flat_imgs_lines_numpy))

    result_lines = inferencer_htr.predict(flat_imgs_lines_numpy)

    PostProcessTranscription.add_trans_to_result(result_full, result_lines)

    for res in result_full:
        for nested_res in res.nested_results:
            for text in nested_res.texts:
                print(text.text)

            print("\n\n")


if __name__ == "__main__":
    region_model = OpenmmlabModelLoader.from_pretrained("Riksarkivet/rtmdet_regions", cache_dir="./models")
    lines_model = OpenmmlabModelLoader.from_pretrained("Riksarkivet/rtmdet_lines", cache_dir="./models")
    text_rec_model = OpenmmlabModelLoader.from_pretrained("Riksarkivet/satrn_htr", cache_dir="./models")

    # inferencer_regions = MMDetInferencer(region_model=region_model)
    # inferencer_lines = MMDetInferencer(region_model=lines_model)
    # inferencer_htr = MMOCRInferencer(text_rec_model=text_rec_model)

    # imgs = glob(
    #     os.path.join(
    #         "/media/erik/Elements/Riksarkivet/data/datasets/htr/Trolldomskommissionen3/Kommissorialrätt_i_Stockholm_ang_trolldomsväsendet,_nr_4_(1676)",
    #         "**",
    #         "bin_image",
    #         "*",
    #     ),
    #     recursive=True,
    # )
    # imgs_numpy = []

    # for img in imgs[0:8]:
    #     imgs_numpy.append(mmcv.imread(img))

    # predict_batch(inferencer_regions, inferencer_lines, inferencer_htr, imgs_numpy)

    # # print(result[-1].img_shape)
    # # print(result['predictions'][0].pred_instances.metadata_fields)
    # # print(result['predictions'][0]._metainfo_fields)
    # # print(result.keys())
    # # print(result)

    # # load image from the IAM database
    # # image = Image.open("./image_0.png").convert("RGB")

    # # Use a pipeline as a high-level helper
    # # from transformers import pipeline

    # # pipe = pipeline("image-to-text", model="microsoft/trocr-large-handwritten")
    # # print(pipe(image, batch_size=8))
