import pytest


pytest.importorskip("kraken", reason="kraken is not installed")

from kraken.containers import BaselineLine, Segmentation  # noqa: E402
from kraken.containers import Region as KrakenRegion

from htrflow.document import Document  # noqa: E402
from htrflow.models.kraken.postprocess import merge_fragments, postprocess_segmentation  # noqa: E402
from htrflow.models.kraken.segmentation import segmentation_to_regions  # noqa: E402


DEMO_IMAGE = "examples/images/pages/A0068699_00021.jpg"


def line(id_, y, x0=100, x1=900, height=40, region=None):
    """A rectangular line with a straight baseline, in page coordinates."""
    return BaselineLine(
        id=id_,
        baseline=[(x0, y), (x1, y)],
        boundary=[(x0, y - height), (x1, y - height), (x1, y), (x0, y)],
        regions=[region] if region else None,
    )


def segmentation(lines, regions=None):
    return Segmentation(
        type="baselines",
        imagename="test.jpg",
        text_direction="horizontal-lr",
        script_detection=False,
        lines=lines,
        regions=regions or {},
    )


@pytest.fixture
def image():
    from PIL import Image

    return Image.open(DEMO_IMAGE)


@pytest.mark.kraken
def test_lines_are_nested_under_a_region(image):
    """Both XML serializers only emit text from two-level trees."""
    region = KrakenRegion(id="r1", boundary=[(50, 50), (950, 50), (950, 500), (50, 500)])
    seg = segmentation([line("l1", 200, region="r1"), line("l2", 300, region="r1")],
                       {"text": [region]})

    regions = segmentation_to_regions(seg, image)

    assert len(regions) == 1
    assert len(regions[0].regions) == 2


@pytest.mark.kraken
def test_lines_without_a_region_are_not_dropped(image):
    """A model that reports no regions must still produce a usable tree."""
    seg = segmentation([line("l1", 200), line("l2", 300)])

    regions = segmentation_to_regions(seg, image)

    assert len(regions) == 1
    assert len(regions[0].regions) == 2


@pytest.mark.kraken
def test_line_coordinates_round_trip(image):
    """
    Line polygons are stored relative to their region, so adding the region
    origin back has to recover the original page coordinates exactly.
    """
    region = KrakenRegion(id="r1", boundary=[(50, 50), (950, 50), (950, 500), (50, 500)])
    lines = [line("l1", 200, region="r1"), line("l2", 300, region="r1")]
    seg = segmentation(lines, {"text": [region]})

    regions = segmentation_to_regions(seg, image)
    origin = regions[0].polygon.bbox.p1

    for node, original in zip(regions[0].regions, lines):
        absolute = [(x, y) for x, y in node.polygon.move(origin)]
        assert absolute == [(int(x), int(y)) for x, y in original.boundary]


@pytest.mark.kraken
def test_region_is_expanded_to_cover_its_lines(image):
    """
    A line reaching outside its region would be cropped against a black
    background, because HTRflow crops the region before cropping the line.
    """
    tight = KrakenRegion(id="r1", boundary=[(100, 100), (900, 100), (900, 250), (100, 250)])
    overhanging = line("l1", 400, region="r1")  # well below the region
    seg = segmentation([overhanging], {"text": [tight]})

    regions = segmentation_to_regions(seg, image)
    region_bbox = regions[0].polygon.bbox
    line_bbox = regions[0].regions[0].polygon.bbox

    assert line_bbox.xmin >= 0 and line_bbox.ymin >= 0
    assert line_bbox.xmax <= region_bbox.width
    assert line_bbox.ymax <= region_bbox.height


@pytest.mark.kraken
def test_baseline_is_kept_for_the_recognition_step(image):
    seg = segmentation([line("l1", 200)])

    node = segmentation_to_regions(seg, image)[0].regions[0]

    assert node.annotations["baseline"] == [(100, 200), (900, 200)]
    assert node.annotations["kraken_id"] == "l1"


@pytest.mark.kraken
def test_reading_order_is_preserved(image):
    """Kraken orders its own lines; the tree must not reshuffle them."""
    region = KrakenRegion(id="r1", boundary=[(50, 50), (950, 50), (950, 900), (50, 900)])
    lines = [line(f"l{i}", 100 + i * 60, region="r1") for i in range(6)]
    seg = segmentation(lines, {"text": [region]})

    regions = segmentation_to_regions(seg, image)

    assert [n.annotations["kraken_id"] for n in regions[0].regions] == [f"l{i}" for i in range(6)]


@pytest.mark.kraken
def test_merge_fragments_joins_a_split_line():
    """Two halves of one physical line, side by side at the same height."""
    left = [(100, 200), (400, 200)]
    right = [(430, 201), (900, 200)]

    merged = merge_fragments([left, right], line_height=40, max_gap=1.5, dy_tol=0.6)

    assert len(merged) == 1
    assert sorted(merged[0][1]) == [0, 1]


@pytest.mark.kraken
def test_merge_fragments_leaves_stacked_lines_alone():
    """Separate lines overlap heavily in x and must never be merged."""
    upper = [(100, 200), (900, 200)]
    lower = [(100, 260), (900, 260)]

    merged = merge_fragments([upper, lower], line_height=40, max_gap=1.5, dy_tol=0.6)

    assert len(merged) == 2


@pytest.mark.kraken
def test_postprocess_drops_short_fragments(image):
    """A stray mark far shorter than the page median is removed."""
    lines = [line(f"l{i}", 100 + i * 60, x0=100, x1=900) for i in range(5)]
    lines.append(line("stray", 500, x0=100, x1=140, height=20))
    seg = segmentation(lines)

    result = postprocess_segmentation(seg, image, repolygonize=False)

    assert len(result.lines) == 5
    assert "stray" not in [ln.id for ln in result.lines]


@pytest.mark.kraken
def test_postprocess_leaves_a_clean_segmentation_alone(image):
    lines = [line(f"l{i}", 100 + i * 60) for i in range(5)]
    seg = segmentation(lines)

    result = postprocess_segmentation(seg, image, repolygonize=False)

    assert len(result.lines) == 5


@pytest.mark.kraken
def test_document_tree_serializes_to_page_xml(image):
    """End-to-end through the real serializer, without loading a model."""
    from htrflow.document import Text
    from htrflow.serialization import get_serializer

    region = KrakenRegion(id="r1", boundary=[(50, 50), (950, 50), (950, 500), (50, 500)])
    seg = segmentation([line("l1", 200, region="r1")], {"text": [region]})

    document = Document(DEMO_IMAGE)
    for node in segmentation_to_regions(seg, image):
        node.attach(document)
    Text("hello", confidence=0.9).attach(document.regions[0].regions[0])

    xml = get_serializer("page").serialize(document)

    assert "hello" in xml
    assert "TextLine" in xml
