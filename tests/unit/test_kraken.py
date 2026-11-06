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
    seg = segmentation([line("l1", 200, region="r1"), line("l2", 300, region="r1")], {"text": [region]})

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


# --- model resolution ------------------------------------------------------
# These need no model files, only kraken on the path.


@pytest.mark.kraken
def test_bundled_segmentation_model_resolves():
    """Omitting `model` falls back to the blla model shipped inside kraken."""
    from htrflow.models.kraken.resolve import resolve_kraken_model

    path, version = resolve_kraken_model(None)

    assert path.endswith("blla.mlmodel")
    assert version == "kraken-bundled"


@pytest.mark.kraken
@pytest.mark.parametrize(
    "model",
    [
        "no/such/model.mlmodel",
        "./missing.safetensors",
        "/absolute/missing/model.mlmodel",
        "a/b/c/deep/path",
    ],
)
def test_a_mistyped_path_is_reported_as_a_path(model):
    """
    A typo'd local path must not come back as a Hugging Face repo-ID
    complaint, which is what happens if it falls through to the HF branch.
    """
    from htrflow.models.kraken.resolve import resolve_kraken_model

    with pytest.raises(FileNotFoundError, match="looks like a file path"):
        resolve_kraken_model(model)


@pytest.mark.kraken
def test_directory_without_weights_is_reported(tmp_path):
    from htrflow.models.kraken.resolve import resolve_kraken_model

    with pytest.raises(FileNotFoundError, match="No kraken model file"):
        resolve_kraken_model(str(tmp_path))


@pytest.mark.kraken
@pytest.mark.parametrize(
    "model,expected",
    [
        ("./my-model.mlmodel", True),
        ("~/models/x.safetensors", True),
        ("/opt/models/x", True),
        ("a/b/c", True),
        ("Riksarkivet/some-kraken-model", False),
        ("catmus-medieval", False),
    ],
)
def test_path_versus_repo_id_heuristic(model, expected):
    from htrflow.models.kraken.resolve import _looks_like_a_path

    assert _looks_like_a_path(model) is expected


@pytest.mark.kraken
def test_a_home_relative_path_is_expanded(tmp_path, monkeypatch):
    """`~/models/x.mlmodel` is how people write paths in YAML."""
    from htrflow.models.kraken.resolve import resolve_kraken_model

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # what Windows reads instead
    (tmp_path / "model.mlmodel").touch()

    assert resolve_kraken_model("~/model.mlmodel") == (str(tmp_path / "model.mlmodel"), "local")


@pytest.mark.kraken
def test_a_missing_absolute_path_is_not_called_relative(tmp_path):
    from htrflow.models.kraken.resolve import resolve_kraken_model

    missing = tmp_path / "missing.mlmodel"
    with pytest.raises(FileNotFoundError) as error:
        resolve_kraken_model(str(missing))

    assert str(missing) in str(error.value)
    assert "relative to" not in str(error.value)


# --- graceful degradation --------------------------------------------------


@pytest.mark.kraken
def test_empty_segmentation_produces_no_regions(image):
    """A page with no detected lines must not raise."""
    assert segmentation_to_regions(segmentation([]), image) == []


@pytest.mark.kraken
def test_postprocess_passes_through_an_empty_segmentation(image):
    seg = segmentation([])

    result = postprocess_segmentation(seg, image)

    assert result.lines == []


@pytest.mark.kraken
def test_postprocess_skips_bbox_segmentation(image):
    """Post-processing is baseline-specific and must decline bbox input."""
    from dataclasses import replace

    seg = replace(segmentation([line("l1", 200)]), type="bbox")

    result = postprocess_segmentation(seg, image)

    assert result is seg


# --- error messages --------------------------------------------------------
# These are user-facing and published, so their wording is worth pinning.


@pytest.mark.kraken
def test_wrong_task_message_names_both_steps():
    """The reader has to be able to tell which YAML key to fix."""
    from htrflow.models.kraken.resolve import wrong_task_message

    message = wrong_task_message("/nonexistent/x.mlmodel", "owner/x", "recognition")

    assert "KrakenSegmentation" in message
    assert "KrakenTextRecognition" in message


@pytest.mark.kraken
def test_wrong_task_message_reads_correctly_when_the_task_is_unknown():
    """An unloadable file must not produce 'is a unknown model'."""
    from htrflow.models.kraken.resolve import wrong_task_message

    message = wrong_task_message("/nonexistent/x.mlmodel", "owner/x", "recognition")

    assert "could not be loaded as a recognition model" in message
    assert "unknown" not in message


@pytest.mark.kraken
def test_resolution_errors_do_not_leak_huggingface_wording():
    """
    A user who mistyped a path should not be told about repo-ID formats, and
    a user who mistyped a repo should not get a raw HTTP error.
    """
    from htrflow.models.kraken.resolve import _unresolvable

    for message in (_unresolvable("owner/typo"),):
        assert "Repo id must be" not in message
        assert "404" not in message
        assert ".safetensors" in message and ".mlmodel" in message


@pytest.mark.kraken
def test_the_working_directory_in_errors_is_the_real_one(tmp_path, monkeypatch):
    """
    The message quotes the working directory to help the reader resolve a
    relative path. It has to be read at call time, not baked in.
    """
    from htrflow.models.kraken.resolve import _unresolvable

    monkeypatch.chdir(tmp_path)

    assert str(tmp_path) in _unresolvable("owner/model")


@pytest.mark.kraken
def test_module_source_contains_no_hardcoded_absolute_paths():
    """
    Guards against paths from a developer's own machine ending up in the
    published docstrings and error messages.
    """
    from pathlib import Path

    import htrflow.models.kraken as package

    for source_file in Path(package.__file__).parent.glob("*.py"):
        source = source_file.read_text()
        for forbidden in ("/home/", "/Users/", "C:\\"):
            assert forbidden not in source, f"{source_file.name} contains {forbidden!r}"


@pytest.mark.kraken
def test_recognition_batches_by_default():
    """
    kraken pads each batch to its widest line, so a batch of one is never
    padded. Recognizers trained on padded lines lose accuracy that way, and
    kraken's own default of 1 is the worst case.
    """
    from htrflow.models.kraken.recognition import KrakenRecognition

    # Called unbound against a stand-in carrying only `device`, so that no
    # model has to be loaded.
    settings = KrakenRecognition._inference_settings(_FakeDevice(), {})

    assert settings["batch_size"] == 32
    assert settings["num_line_workers"] == 0


@pytest.mark.kraken
def test_inference_settings_are_overridable():
    from htrflow.models.kraken.recognition import KrakenRecognition

    settings = KrakenRecognition._inference_settings(_FakeDevice(), {"batch_size": 4, "num_line_workers": 2})

    assert settings["batch_size"] == 4
    assert settings["num_line_workers"] == 2


class _FakeDevice:
    """Stand-in exposing just the `device` attribute `_inference_settings` reads."""

    class device:
        type = "cpu"
        index = None


# --- the recognition step --------------------------------------------------
# How the step reads lines out of the document tree and matches the
# transcriptions back. The recognizer is a stand-in, so no weights are needed.


def _document(regions):
    """The demo page with `regions` attached."""
    document = Document(DEMO_IMAGE)
    for region in regions:
        region.attach(document)
    return document


def _collect(document):
    """`_collect_lines` against the document's own page image."""
    from htrflow.models.kraken.step import _collect_lines

    return _collect_lines(document, document.image.size)


def _box(x0, y0, x1, y1):
    from htrflow.utils.geometry import Polygon

    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _record(id_, prediction, confidences=(0.9, 0.7)):
    """A stand-in for kraken's `BaselineOCRRecord`."""
    from types import SimpleNamespace

    return SimpleNamespace(id=id_, prediction=prediction, confidences=list(confidences))


class _FakeRecognizer:
    """Stand-in for `KrakenRecognition` that returns canned records."""

    metadata = {"model": "fake"}

    def __init__(self, records_for):
        self.records_for = records_for

    def recognize(self, image, segmentation, **config_kwargs):
        return self.records_for(segmentation)


class _FakeSegmenter:
    """Stand-in for kraken's `SegmentationTaskModel` that returns a fixed result."""

    def __init__(self, result):
        self.result = result

    def predict(self, image, config):
        return self.result


@pytest.mark.kraken
def test_step_hands_kraken_back_its_own_lines(image):
    """
    Segmentation stores lines relative to their region. The step has to undo
    that exactly, or kraken extracts the wrong strip of the page.
    """

    region = KrakenRegion(id="r1", boundary=[(250, 300), (1600, 300), (1600, 800), (250, 800)])
    lines = [line(f"l{i}", 400 + 80 * i, x0=300, x1=1500, region="r1") for i in range(4)]
    seg = segmentation(lines, {"text": [region]})

    _, collected = _collect(_document(segmentation_to_regions(seg, image)))

    assert [ln.id for ln in collected] == [ln.id for ln in lines]
    assert [ln.baseline for ln in collected] == [ln.baseline for ln in lines]
    assert [ln.boundary for ln in collected] == [ln.boundary for ln in lines]


@pytest.mark.kraken
def test_step_reads_lines_placed_directly_on_the_page():
    """A single-stage line segmenter attaches its lines straight to the page."""
    from htrflow.document import Region

    document = _document([Region(_box(100, y - 40, 900, y)) for y in (200, 300)])

    nodes, collected = _collect(document)

    assert nodes == document.regions
    for y, record in zip((200, 300), collected):
        assert record.boundary == [(100, y - 40), (900, y - 40), (900, y), (100, y)]
        # No baseline was stored, so one is synthesised inside the box.
        assert all(100 <= x <= 900 and y - 40 <= by <= y for x, by in record.baseline)


@pytest.mark.kraken
def test_step_adds_up_offsets_through_deeper_trees():
    """Each level stores its polygon relative to its parent's bounding box."""
    from htrflow.document import Region

    column = Region(_box(100, 200, 900, 1000))  # page coordinates
    paragraph = Region(_box(50, 100, 750, 500))  # relative to the column
    Region(_box(10, 20, 600, 60)).attach(paragraph)  # relative to the paragraph
    paragraph.attach(column)

    _, (collected,) = _collect(_document([column]))

    # 100 + 50 + 10 = 160 and 200 + 100 + 20 = 320
    assert collected.boundary == [(160, 320), (750, 320), (750, 360), (160, 360)]


@pytest.mark.kraken
def test_step_reads_an_unsegmented_page_as_one_line():
    """As the generic `TextRecognition` does, for example on line images."""

    document = Document(DEMO_IMAGE)
    width, height = document.image.size

    nodes, (collected,) = _collect(document)

    assert nodes == [document]
    # Clamped onto the last column and row, as kraken requires.
    assert {x for x, _ in collected.boundary} == {0, width - 1}
    assert {y for _, y in collected.boundary} == {0, height - 1}


@pytest.mark.kraken
def test_step_keeps_lines_inside_the_image():
    """
    kraken refuses a line with any point outside the image, and returns an
    empty transcription for it instead of raising.
    """
    from htrflow.document import Region

    document = Document(DEMO_IMAGE)
    width, height = document.image.size
    Region(_box(-20, height - 50, width + 30, height + 10)).attach(document)

    _, (collected,) = _collect(document)

    assert all(0 <= x < width and 0 <= y < height for x, y in collected.boundary + collected.baseline)


@pytest.mark.kraken
def test_step_skips_inactive_lines(image):
    from htrflow.document import InactiveRegion

    region = KrakenRegion(id="r1", boundary=[(50, 50), (950, 50), (950, 500), (50, 500)])
    seg = segmentation([line("l1", 200, region="r1"), line("l2", 300, region="r1")], {"text": [region]})
    document = _document(segmentation_to_regions(seg, image))
    InactiveRegion().attach(document.regions[0].regions[0])

    _, collected = _collect(document)

    assert [ln.id for ln in collected] == ["l2"]


@pytest.mark.kraken
def test_a_page_without_lines_is_marked_inactive(image):
    """
    Otherwise the empty page stays a leaf, and recognition would read the
    whole page as a single line.
    """
    import torch

    from htrflow.document import NoRegionsDetected
    from htrflow.models.kraken.segmentation import KrakenSegmentation

    model = KrakenSegmentation.__new__(KrakenSegmentation)  # skips loading weights
    model.device = torch.device("cpu")
    model.model = _FakeSegmenter(segmentation([]))

    (results,) = model._predict([image])
    document = _document(results)

    assert [type(result) for result in results] == [NoRegionsDetected]
    assert not document.is_active
    assert _collect(document) == ([], [])


@pytest.mark.kraken
def test_segmentation_is_kraken_own_unless_postprocessing_is_asked_for(image):
    """Post-processing changes kraken's output, so it has to be opted into."""
    import torch

    from htrflow.models.kraken.segmentation import KrakenSegmentation

    lines = [line(f"l{i}", 100 + i * 60) for i in range(5)]
    lines.append(line("stray", 500, x0=100, x1=140, height=20))
    model = KrakenSegmentation.__new__(KrakenSegmentation)  # skips loading weights
    model.device = torch.device("cpu")
    model.model = _FakeSegmenter(segmentation(lines))

    def kraken_ids(regions):
        return {node.annotations["kraken_id"] for region in regions for node in region.regions}

    (default,) = model._predict([image])
    (cleaned,) = model._predict([image], postprocess=True, postprocess_settings={"repolygonize": False})

    assert "stray" in kraken_ids(default)
    assert "stray" not in kraken_ids(cleaned)


@pytest.mark.kraken
def test_step_matches_transcriptions_to_lines_by_id(image):
    """
    If kraken ever returned fewer records than lines, or in another order,
    matching by position would put every later transcription on the wrong
    line.
    """
    from htrflow.models.kraken.step import KrakenTextRecognition

    region = KrakenRegion(id="r1", boundary=[(50, 50), (950, 50), (950, 700), (50, 700)])
    seg = segmentation([line(f"l{i}", 200 + 100 * i, region="r1") for i in range(3)], {"text": [region]})
    document = _document(segmentation_to_regions(seg, image))

    def records_for(segmentation):
        # Out of order, the middle line missing, and an id the step never sent.
        ids = [ln.id for ln in segmentation.lines]
        return [_record(ids[2], "third"), _record(ids[0], "first"), _record("stray", "?")]

    KrakenTextRecognition(_FakeRecognizer(records_for)).run(document)

    nodes = document.regions[0].regions
    assert [[text.text for text in node.transcription] for node in nodes] == [["first"], [], ["third"]]
    assert nodes[0].transcription[0].confidence == pytest.approx(0.8)


@pytest.mark.kraken
def test_line_confidence_is_the_mean_and_never_none():
    """`Text.attach` sorts transcriptions by confidence, which fails on None."""
    from types import SimpleNamespace

    from htrflow.models.kraken.step import _confidence

    assert _confidence(SimpleNamespace(confidences=[0.5, 1.0])) == 0.75
    assert _confidence(SimpleNamespace(confidences=[])) == 0.0
    assert _confidence(SimpleNamespace()) == 0.0


@pytest.mark.kraken
def test_step_reports_lines_that_come_back_empty(image, caplog):
    """An empty record is the only sign kraken gives of a failed line."""
    from htrflow.models.kraken.step import KrakenTextRecognition

    document = _document(segmentation_to_regions(segmentation([line("l1", 200), line("l2", 300)]), image))

    def records_for(segmentation):
        first, second = (ln.id for ln in segmentation.lines)
        return [_record(first, "text"), _record(second, "", confidences=())]

    with caplog.at_level("WARNING", logger="htrflow.models.kraken.step"):
        KrakenTextRecognition(_FakeRecognizer(records_for)).run(document)

    assert "1 of 2 lines" in caplog.text
    assert "came back empty" in caplog.text


@pytest.mark.kraken
def test_fallback_line_box_stays_inside_the_image():
    """Near the top of the page, the padding would otherwise reach above it."""
    from htrflow.models.kraken.postprocess import _box_around

    box = _box_around([(10, 5), (990, 8)], line_height=40, size=(1000, 800))

    assert all(0 <= x < 1000 and 0 <= y < 800 for x, y in box)
    assert min(y for _, y in box) == 0


# --- step registration -----------------------------------------------------


@pytest.mark.kraken
def test_pipelines_find_the_kraken_step():
    from htrflow.models.kraken.step import KrakenTextRecognition
    from htrflow.pipeline.steps import get_step_class

    assert get_step_class("KrakenTextRecognition") is KrakenTextRecognition
    assert get_step_class("krakentextrecognition") is KrakenTextRecognition


@pytest.mark.kraken
def test_the_step_module_can_be_imported_first():
    """
    The step module and `htrflow.pipeline.steps` used to import each other at
    module level, so importing the step module first failed with a circular
    import. The test process has imported both already, so the check runs in
    a fresh interpreter.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    import htrflow.models.kraken as package

    # Make the child import this same checkout rather than an installed copy.
    src = str(Path(package.__file__).parents[3])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")])))
    code = (
        "import htrflow.models.kraken.step\n"
        "from htrflow.pipeline.steps import get_step_class\n"
        "print(get_step_class('KrakenTextRecognition').__name__)\n"
    )

    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)

    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip().splitlines()[-1] == "KrakenTextRecognition"


@pytest.mark.kraken
def test_a_missing_optional_dependency_names_its_extra(monkeypatch):
    """A pipeline asking for an optional step should say what to install."""
    from htrflow.pipeline import steps

    monkeypatch.setitem(steps.OPTIONAL_STEPS, "UninstalledStep", ("htrflow.no_such_module", "someextra"))

    with pytest.raises(ModuleNotFoundError, match=r"pip install htrflow\[someextra\]"):
        steps.get_step_class("UninstalledStep")


# --- settings for other material and models --------------------------------


class _FakeNet:
    """Records the width of every batch kraken would hand the network."""

    def __init__(self):
        self.widths = []

    def _rec_predict(self, seqs, seq_lens):
        self.widths.append(seqs.shape[-1])
        return None, None


class _FakeTaskModel:
    """Stand-in for kraken's `RecognitionTaskModel` that runs one batch."""

    def __init__(self, net):
        self.net = net

    def predict(self, image, segmentation, config):
        import torch

        self.net._rec_predict(torch.zeros(2, 3, 96, 100), torch.tensor([100, 80]))
        return iter([])


@pytest.mark.kraken
def test_pad_to_width_pads_every_batch_while_recognizing():
    """Models trained on fixed-width lines are more accurate with the same padding."""
    import torch

    from htrflow.models.kraken.recognition import KrakenRecognition, _batches_padded_to

    net = _FakeNet()
    model = KrakenRecognition.__new__(KrakenRecognition)  # skips loading weights
    model.device = torch.device("cpu")
    model.model = _FakeTaskModel(net)

    model.recognize(None, None, pad_to_width=2560)
    model.recognize(None, None)
    with _batches_padded_to(net, 2560):
        net._rec_predict(torch.zeros(1, 3, 96, 3000), torch.tensor([3000]))  # wider than the target

    assert net.widths == [2560, 100, 3000]
    assert "_rec_predict" not in vars(net)  # the model is left as it was


@pytest.mark.kraken
def test_pad_to_width_reaches_kraken_s_real_network(tmp_path):
    """
    `pad_to_width` works through `_rec_predict`, a method internal to kraken
    that a kraken release could change without notice. The test above uses a
    stand-in, so this one runs kraken's real loading and recognition code on
    a real PP-OCRv6 network, untrained so that nothing has to be downloaded.
    If a kraken release stops the padding from reaching the network, this
    test fails rather than users' transcriptions quietly getting worse.
    """
    import torch
    from kraken.lib.ppocr.model import PPOCRv6Model
    from kraken.models.writers import write_safetensors
    from PIL import Image

    from htrflow.models.kraken.recognition import KrakenRecognition

    path = tmp_path / "untrained.safetensors"
    write_safetensors([PPOCRv6Model(variant="tiny", num_classes=3, codec={"a": [1], "b": [2]})], path)
    model = KrakenRecognition(str(path), device="cpu")

    # Lines with ink in them: kraken never sends a blank line to the network.
    page = Image.open(DEMO_IMAGE)
    lines = segmentation([line("long", 300), line("short", 500, x1=500)])

    # Record the width of every image batch any layer receives. The widest is
    # the batch kraken hands the network.
    widths = []

    def record_width(module, args):
        if args and isinstance(args[0], torch.Tensor) and args[0].ndim == 4:
            widths.append(args[0].shape[-1])

    hook = torch.nn.modules.module.register_module_forward_pre_hook(record_width)
    try:
        unpadded = model.recognize(page, lines)
        widest_unpadded = max(widths)
        widths.clear()
        padded = model.recognize(page, lines, pad_to_width=2560)
        widest_padded = max(widths)
    finally:
        hook.remove()

    assert widest_unpadded < 2560  # by default, only as wide as the widest line
    assert widest_padded == 2560
    assert len(unpadded) == len(padded) == 2
    assert "_rec_predict" not in vars(model.model.net)  # the model is left as it was


@pytest.mark.kraken
@pytest.mark.parametrize(
    "device,expected",
    [
        ("cpu", ("cpu", 1)),
        ("cuda", ("gpu", 1)),
        ("cuda:1", ("gpu", [1])),
        ("mps", ("mps", 1)),
    ],
)
def test_torch_devices_map_onto_kraken_accelerators(device, expected):
    import torch

    from htrflow.models.kraken.recognition import lightning_settings

    settings = lightning_settings(torch.device(device), {})

    assert (settings["accelerator"], settings["device"]) == expected


@pytest.mark.kraken
def test_explicit_accelerator_settings_win():
    import torch

    from htrflow.models.kraken.recognition import lightning_settings

    settings = lightning_settings(torch.device("cuda"), {"accelerator": "cpu", "device": 1})

    assert (settings["accelerator"], settings["device"]) == ("cpu", 1)


@pytest.mark.kraken
def test_lines_carry_per_character_scores():
    """Other HTRflow recognizers report token scores, which later steps read."""
    from htrflow.models.kraken.step import _token_scores

    assert _token_scores(_record("ab", "ab", confidences=(0.9, 0.5))) == [("a", 0.9), ("b", 0.5)]
    # A label that decodes to several code points breaks the pairing.
    assert _token_scores(_record("ab", "abc", confidences=(0.9, 0.5))) == []


@pytest.mark.kraken
def test_postprocess_leaves_vertical_text_alone(image):
    """The merge heuristics assume horizontal lines."""
    from dataclasses import replace

    seg = replace(segmentation([line("l1", 200), line("stray", 300, x0=100, x1=110)]), text_direction="vertical-rl")

    assert postprocess_segmentation(seg, image) is seg
