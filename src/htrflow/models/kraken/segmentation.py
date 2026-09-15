"""
HTRflow adapter of kraken's baseline segmenter.
"""

import logging
from typing import Any

from PIL import Image

from htrflow.document import Region
from htrflow.models.base_model import BaseModel
from htrflow.models.kraken.postprocess import postprocess_segmentation
from htrflow.models.kraken.resolve import resolve_kraken_model
from htrflow.utils.geometry import Polygon


logger = logging.getLogger(__name__)


class KrakenSegmentation(BaseModel):
    """
    HTRflow adapter of kraken's baseline segmentation models.

    Unlike the YOLO segmenter, which needs one pass for regions and a second
    pass for lines within each region, kraken returns regions and lines from
    a single pass and assigns each line to its region. This model therefore
    produces a complete two-level tree in one `Segmentation` step.

    kraken also returns its lines in reading order, so **do not** follow this
    step with `ReadingOrderMarginalia` or `OrderLines` -- both would re-sort
    the lines and discard the ordering kraken established.

    Each line carries its baseline in `Region.annotations`, which is what
    `KrakenTextRecognition` uses to hand kraken back a segmentation it can
    extract dewarped line images from.

    Example usage with the `Segmentation` step:
    ```yaml
    - step: Segmentation
      settings:
        model: KrakenSegmentation
        model_settings:
          model: validation/model/kraken_seg_model.mlmodel
          device: cuda
        generation_settings:
          text_direction: horizontal-lr
          postprocess: true
          postprocess_settings:
            min_len_frac: 0.12
            merge_gap: 1.5
            dy_tol: 0.6
    ```

    `model` accepts a local path, an htrmopo DOI, a Hugging Face repo ID, or
    may be omitted to use the segmentation model bundled with kraken.
    """

    def __init__(self, model: str | None = None, revision: str | None = None, **kwargs):
        """
        Arguments:
            model: Path to a kraken segmentation model, an htrmopo DOI, or a
                Hugging Face repo ID. Defaults to kraken's bundled
                `blla.mlmodel`.
            revision: Optional revision of the Hugging Face repository.
            **kwargs: Forwarded to `BaseModel.__init__`, notably `device`.
        """
        super().__init__(**kwargs)

        from kraken.tasks import SegmentationTaskModel

        model_path, model_version = resolve_kraken_model(model, revision)
        self.model = SegmentationTaskModel.load_model(model_path)

        logger.info(
            "Initialized kraken segmentation model '%s' from %s on device %s",
            model, model_path, self.device,
        )
        self.metadata.update({"model": model or "blla", "model_version": model_version})

    def _predict(
        self,
        images: list[Image.Image],
        postprocess: bool = True,
        postprocess_settings: dict[str, Any] | None = None,
        **config_kwargs,
    ) -> list[list[Region]]:
        """
        Run kraken segmentation and convert the result to HTRflow regions.

        Arguments:
            images: Input images.
            postprocess: Whether to run `postprocess.postprocess_segmentation`
                on kraken's output. Recommended for fine-tuned handwriting
                models, which tend to fragment lines.
            postprocess_settings: Keyword arguments for
                `postprocess.postprocess_segmentation`.
            **config_kwargs: Forwarded to kraken's
                `SegmentationInferenceConfig`, for example `text_direction`
                or `precision`.
        """
        from kraken.configs import SegmentationInferenceConfig

        config = SegmentationInferenceConfig(**self._device_settings(config_kwargs))

        results = []
        for image in images:
            segmentation = self.model.predict(image, config)
            if postprocess:
                segmentation = postprocess_segmentation(
                    segmentation, image, **(postprocess_settings or {})
                )
            results.append(segmentation_to_regions(segmentation, image))
        return results

    def _device_settings(self, config_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Translate HTRflow's torch device into kraken's Lightning settings.

        kraken configures inference through a Lightning `Fabric`, which takes
        an accelerator plus device indices rather than a torch device. Any
        explicit `accelerator` or `device` in the YAML wins.
        """
        settings = dict(config_kwargs)
        if "accelerator" not in settings:
            settings["accelerator"] = "gpu" if self.device.type == "cuda" else "cpu"
        if "device" not in settings:
            settings["device"] = [self.device.index] if self.device.index is not None else 1
        return settings


def segmentation_to_regions(segmentation, image: Image.Image) -> list[Region]:
    """
    Convert a kraken `Segmentation` into a two-level HTRflow region tree.

    Both XML serializers only emit text from `page -> region -> line` trees,
    so lines are always nested inside a region, and a page-sized region is
    synthesised when kraken reports none.

    Line polygons are translated into their region's local coordinate frame,
    which is the convention HTRflow's image loader and templates expect. Each
    region is expanded to cover its lines, because a child polygon reaching
    outside its parent's bounding box would otherwise be cropped against a
    black background rather than the page.

    Arguments:
        segmentation: A kraken segmentation.
        image: The page it was computed on, used to size a synthesised region.

    Returns:
        Top-level regions in reading order, each holding its lines.
    """
    regions_by_id = {
        region.id: region
        for regions in (segmentation.regions or {}).values()
        for region in regions
    }

    # Order regions by the first line that belongs to them, so region order
    # follows the reading order kraken established for the lines.
    order: list[str] = []
    for line in segmentation.lines:
        for region_id in line.regions or []:
            if region_id in regions_by_id and region_id not in order:
                order.append(region_id)

    groups = [
        (regions_by_id[region_id].boundary,
         [ln for ln in segmentation.lines if region_id in (ln.regions or [])])
        for region_id in order
    ]

    # Lines kraken did not assign to a region still have to end up somewhere,
    # or the serializers drop them silently. A model that reports no regions
    # at all lands here too.
    orphans = [ln for ln in segmentation.lines if not (set(ln.regions or []) & set(order))]
    if orphans:
        width, height = image.size
        page = [(0, 0), (width, 0), (width, height), (0, height)]
        groups.append((page, orphans))
        logger.info("Placed %d unassigned line(s) in a synthesised page region.", len(orphans))

    regions = []
    for boundary, lines in groups:
        line_boundaries = [ln.boundary or ln.baseline for ln in lines]
        polygon = Polygon(_expand_to_cover(boundary, line_boundaries))
        region = Region(polygon, segmentation_label="region")

        origin = polygon.bbox.p1
        for line, line_boundary in zip(lines, line_boundaries):
            Region(
                Polygon(line_boundary).move(-origin),
                segmentation_label="line",
                baseline=[(int(x), int(y)) for x, y in line.baseline],
                kraken_id=line.id,
            ).attach(region)
        regions.append(region)

    return regions


def _expand_to_cover(boundary, line_boundaries) -> list[tuple[int, int]]:
    """
    Expand a region boundary so that it covers all of its lines.

    kraken's region polygons are derived independently of its line polygons,
    so a line can extend past the region it is assigned to. Since HTRflow
    crops each region and then crops lines out of that crop, any overhang
    would come back as black padding instead of page content.

    Falls back to the bounding box of everything if shapely cannot produce a
    single polygon (which happens when a line sits fully outside its region).
    """
    if not line_boundaries:
        return [(int(x), int(y)) for x, y in boundary]

    from shapely.geometry import MultiPoint
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    shapes = []
    for points in [boundary, *line_boundaries]:
        if len(points) < 3:
            continue
        # buffer(0) repairs the self-intersecting polygons kraken sometimes
        # emits, which would otherwise make the union fail.
        shapes.append(ShapelyPolygon(points).buffer(0))

    if not shapes:
        return [(int(x), int(y)) for x, y in boundary]

    union = unary_union(shapes)
    if union.geom_type != "Polygon":
        # Disjoint pieces: fall back to a hull that covers all of them.
        union = MultiPoint([p for shape in shapes for p in shape.exterior.coords]).convex_hull

    # Shapely repeats the first point to close the ring; HTRflow does not.
    coords = [(int(x), int(y)) for x, y in union.exterior.coords[:-1]]
    return coords or [(int(x), int(y)) for x, y in boundary]
