import os

import pandas as pd
from datasets import load_dataset


# Boiler plate for the general idea of having like a dataframe to populate...


class DataStructures:
    SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".tiff")

    def __init__(self):
        self.df = pd.DataFrame(columns=["Filename", "Filepath"])

    def create_inital_ds(self, image_data_folder):
        for filename in os.listdir(image_data_folder):
            if filename.endswith(self.SUPPORTED_FORMATS):
                filepath = os.path.join(image_data_folder, filename)
                self.df = self.df.append({"Filename": filename, "Filepath": filepath}, ignore_index=True)

    def add_metadata(self, metadata):
        pass

    def load_huggingface_dataset(self, dataset_name):
        hf_dataset = load_dataset(dataset_name)
        df = hf_dataset.to_pandas()
        return df

    # Perhaps new class or extends this with
    def add_to_ds():
        """creates an copy of existing dataframe and adds output from the inferencer that was from (e.g. textseg or textrec..)"""
        pass

    def data_structure_joiner(direction="left"):
        """Here we could join different DataStructure object with each other.."""
        pass


if __name__ == "__main__":
    data_folder = "path/to/image/folder"
    data_structures = DataStructures()
    data_structures.create_inital_ds(data_folder)

    print(data_structures.df.head())
