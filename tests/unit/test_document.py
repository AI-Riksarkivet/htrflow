import pytest

from htrflow.document import Document, ImageLoader, InactiveRegion


@pytest.fixture
def example_doucment():
    return Document("examples/images/pages/A0068699_00021.jpg")


def test_active_region(example_doucment):
    assert example_doucment.is_active


def test_inactive_region(example_doucment):
    inactive = InactiveRegion()
    inactive.attach(example_doucment)
    assert not example_doucment.is_active


def test_image_loader(example_doucment):
    loader = ImageLoader(example_doucment)
    assert len(list(loader)) == 1


def test_image_loader_ignore_inactive(example_doucment):
    example_doucment.is_active = False
    loader = ImageLoader(example_doucment)
    assert len(list(loader)) == 0


def test_image_loader_dont_ignore_inactive(example_doucment):
    example_doucment.is_active = False
    loader = ImageLoader(example_doucment, ignore_inactive=False)
    assert len(list(loader)) == 1
