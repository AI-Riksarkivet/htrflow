#!/usr/bin/env python3
import datasets


_TEST_DESCRIPTION = "This is a test description."

class HTRFlowConfig(datasets.BuilderConfig):
    def __init__(self, features, data_path, citation, url, label_classes=("False", "True"), **kwargs):
        super(HTRFlowConfig, self).__init__(version=datasets.Version("1.0.2"), **kwargs)
        self.features = features
        self.label_classes = label_classes
        self.data_path = data_path
        self.citation = citation
        self.url = url

class HTRFlowData(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = HTRFlowConfig

    BUILDER_CONFIGS = [
        HTRFlowConfig(name="test_cfg", description=_TEST_DESCRIPTION, features={"idx": datasets.Value(dtype="int32", id=None)}, data_path="test/path", citation="test citation", url="test-url.com"),

    ]

    def _info(self):
        {feature: datasets.Value("string") for feature in self.config.features}
        if self.config.name.startswith("test"):
            pass
        return datasets.DatasetInfo(
            description=self.config.description,
            features=datasets.Features()
        )
