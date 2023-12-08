# import os
# from imghdr import what

from datasets import Dataset, Image, load_dataset


# class DatasetCreator:
#     def __init__(self, image_folder=None, dataset_name=None):
#         self.dataset = None
#         if image_folder:
#             self.load_from_folder(image_folder)
#         elif dataset_name:
#             self.load_hf_dataset(dataset_name)

#     def load_from_folder(self, image_folder):
#         images_path_list = []
#         for image in os.listdir(image_folder):
#             full_path = os.path.join(image_folder, image)
#             if what(full_path):
#                 images_path_list.append(full_path)
#         self.dataset = Dataset.from_dict({"image": images_path_list}).cast_column("image", Image())

#     def load_hf_dataset(self, dataset_name):
#         self.dataset = load_dataset(dataset_name, split="train")

#     def get_dataset(self):
#         return self.dataset


class BatchLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = self._validate_object_type_and_return_dataset(dataset)
        self.batch_size = batch_size
        self.index = 0

    def _validate_object_type_and_return_dataset(self, dataset):
        if not isinstance(dataset, Dataset):
            raise TypeError("Data must be of type 'datasets.Dataset'")
        return dataset

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        batch = self.dataset[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return batch




if __name__ == "__main__":


    image_folder = "/home/gabriel/Desktop/htrflow_core/data/raw"

    dataset = load_dataset("imagefolder", data_dir=image_folder, split="train")

    image_dataset= dataset.cast_column("image", Image())

    # dataset_creator = DatasetCreator(image_folder="/path/to/image/folder")
    # dataset_creator = DatasetCreator(dataset_name="scene_parse_150")
    # dataset = dataset_creator.get_dataset()

    print(image_dataset)


    d_ataset = load_dataset("nateraw/ade20k-tiny")
    # # Create and use the batch loader
    batch_loader = BatchLoader(image_dataset , batch_size=2)

    for batch in batch_loader:
        print(batch['image'])  # Process the batch

    batch_loader_2 = BatchLoader(d_ataset['train']  , batch_size=3)

    for batch in batch_loader_2:
        print(batch['image'])  # Process the batch



# TODO Be able to load from folder
# TODO Be able to load from HF
# TODO Be able to stream
# TODO Be able to have seperate datasets_buidler loading script?
# TODO query from datasets-server


