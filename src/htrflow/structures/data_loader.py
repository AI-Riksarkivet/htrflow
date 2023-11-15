import cv2

# TODO: add rapids.ai


class DataLoader:
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


if __name__ == "__main__":
    import pandas as pd

    temp_data_structures_df = pd.DataFrame(columns=["Filename", "Filepath"])

    generator = DataLoader(temp_data_structures_df)
    for image_batch in generator.image_batch_loader(batch_size=10):
        ...  # process image batch
