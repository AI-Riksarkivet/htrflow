
from htrflow.datasets.batch_loader import BatchLoader
from htrflow.inferencers.base_inferencer import BaseInferencer
from htrflow.tasks.base_task import BaseTask


class TextSegmentation(BaseTask):
    def __init__(self, text_seg_inferencer: BaseInferencer):
        self.text_rec_inferencer= text_seg_inferencer

    def preprocess(self):
        pass

    # TODO perhaps the batch prcoess could be moved outside into a taskmangar /batch_task_manager? and that TextSegmentation is passed into it?
    # i.e Textsemgnetion can be run for just an dataset[image].. . However if we use it with batch_task_manager we can do in batch operation?

## test batching with HF map
    def batch_run(self, dataset_of_image , **kwargs):

        output_column = kwargs["output_column"]

        print(len(dataset_of_image["image"]))

        new_images = []

        for image in dataset_of_image["image"]:
            new_images.append(image)

        return {output_column: new_images}

# Test batch with hf map..
    def run(self, dataset_of_image, target_column, output_column):
        import uuid
        new_column = [str(uuid.uuid4()) for _ in range(len(dataset_of_image))]
        dataset_of_image = dataset_of_image.add_column(f"{target_column}_uuid", new_column)

        dataset_of_image_new = dataset_of_image.map(self.batch_run, batched=True, batch_size =3, remove_columns = target_column , fn_kwargs={ "output_column": output_column} )

        print(dataset_of_image_new['image_uuid'] )

        # for batch_of_images in dataset_of_image:
        #     print(batch_of_images[target_column] )


        # standard_dict = self.text_rec_inferencer.predict(input_images)

        # process_standard_dict = self.postprocess(standard_dict)

        # dataset["predictions_images"] = process_standard_dict

        # print(input_images)

    def postprocess(self):
        pass

if __name__ == "__main__":

    from datasets import load_dataset

    from htrflow.inferencers.openmmlab.mmdet_inferencer import MMDetInferencer
    from htrflow.models.openmmlab_loader import OpenmmlabModelLoader

    image_dataset = load_dataset("nateraw/ade20k-tiny", split="train")
    # # Create and use the batch loader
    batch_loader = BatchLoader(image_dataset  , batch_size=2)

    print(image_dataset)

    rtmdet_region_model = OpenmmlabModelLoader.from_pretrained("Riksarkivet/rtmdet_regions", cache_dir="/home/gabriel/Desktop/htrflow_core/models")
    text_segmenter= TextSegmentation(MMDetInferencer(rtmdet_region_model))

    text_segmenter.run(image_dataset, "image", "new_images")
