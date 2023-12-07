import cv2

# TODO: add rapids.ai


class TestDataLoader:
    def __init__(
        self,
        dataframe,
    ):
        self.dataframe = dataframe

    def image_batch_loader(self, batch_size=8):
        batch = []
        for _, row_path in self.dataframe.iterrows():
            image = cv2.imread(row_path["Filepath"])

            batch.append(image)
            if len(batch) == batch_size:
                yield batch
                batch = []

    def image_loader():
        pass

    def polygons_to_mask_loader():
        pass



if __name__ == "__main__":

    import os
    from imghdr import what

    from datasets import Dataset, Image



    image_folder = "/home/gabriel/Desktop/htrflow_core/data/raw"

    images_path_list = []
    for image in os.listdir(image_folder):
        full_path = os.path.join(image_folder, image)
        if what(full_path):
            images_path_list.append(full_path)


    fake_images_path_list = images_path_list*10


    dataset = Dataset.from_dict({"image": fake_images_path_list}).cast_column("image", Image())
    print(dataset)

    class CustomBatchLoader:
        def __init__(self, data, batch_size):
            self.data = data
            self.batch_size = batch_size
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.data):
                raise StopIteration
            batch = self.data[self.index:self.index + self.batch_size]
            self.index += self.batch_size
            return batch

    # Usage
    batch_loader = CustomBatchLoader(dataset, batch_size=8)

    for batch in batch_loader:
        print(len(batch['image'] ))

    # numpy_dataset = dataset.with_format("numpy")


    # dataloader = DataLoader(numpy_dataset, batch_size=3)


    # for image in numpy_dataset:
    #     print(image)

    # print(numpy_dataset.to_pandas())




    # import pandas as pd

    # temp_data_structures_df = pd.DataFrame(columns=["Filename", "Filepath"])

    # generator = DataLoader(temp_data_structures_df)
    # for image_batch in generator.image_batch_loader(batch_size=10):
    #     ...  # process image batch
