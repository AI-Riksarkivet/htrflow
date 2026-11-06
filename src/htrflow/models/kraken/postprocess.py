"""
Clean up kraken baseline segmentation output.

Fine-tuned baseline segmenters tend to make two mistakes on handwritten
material that can be fixed after the fact, without retraining:

    1. One line is found as two fragments, split mid-word, so that a word
       becomes a one-letter fragment plus the rest.
       Fix: join fragments that sit side by side at the same height.
    2. Short spurious baselines are found on single letters or ascenders,
       between the real lines.
       Fix: drop baselines far shorter than the page's median baseline.

Both fixes change the baselines, so the line polygons kraken derived from
the old ones no longer fit. They are therefore derived again from the
corrected baselines, with kraken's `calculate_polygonal_environment`.

Joining comes before dropping, so that a real line split into a tiny
fragment is rejoined before the length test can discard it.

Text the segmenter missed entirely cannot be recovered this way.
"""

import logging
import math
import uuid
from dataclasses import replace
from statistics import median


logger = logging.getLogger(__name__)


def baseline_length(baseline) -> float:
    """Total length of a polyline, in pixels."""
    return sum(math.dist(baseline[i], baseline[i + 1]) for i in range(len(baseline) - 1))


def _x_extent(baseline) -> tuple[float, float]:
    xs = [point[0] for point in baseline]
    return min(xs), max(xs)


def merge_fragments(
    baselines: list[list[tuple[int, int]]],
    line_height: float,
    max_gap: float,
    dy_tol: float,
    max_overlap: float = 0.3,
) -> list[tuple[list[tuple[int, int]], list[int]]]:
    """
    Greedily join baseline fragments belonging to the same physical line.

    Two fragments merge when they sit side by side (a small horizontal gap,
    little horizontal overlap) and the end of the left one is at about the
    height of the start of the right one. Separate lines stacked on top of
    each other overlap heavily horizontally, so they are never merged. A
    genuine insertion written between two lines sits about a line height
    higher, so it is not merged either. The closest pair is merged first,
    and merging repeats until no pair qualifies.

    Arguments:
        baselines: Baselines to consider, as lists of (x, y) points.
        line_height: Median line height in pixels, used as the unit for
            `max_gap` and `dy_tol`.
        max_gap: Maximum horizontal gap to bridge, in line heights.
        dy_tol: Maximum vertical mismatch at the junction, in line heights.
        max_overlap: Maximum share of the shorter fragment that may overlap
            the longer one in x before the pair is rejected.

    Returns:
        A list of `(baseline, source_indices)` tuples, where `source_indices`
        names the input baselines that were merged into it. Unmerged
        baselines come back with a single index, so callers can carry over
        tags and region membership from the original line records.
    """
    # Each candidate carries the indices of the input baselines it was built
    # from, so the caller can tell which original lines ended up where.
    candidates = [(sorted(bl, key=lambda p: p[0]), [i]) for i, bl in enumerate(baselines)]

    while True:
        best = None
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i][0], candidates[j][0]
                left, right = (a, b) if a[0][0] <= b[0][0] else (b, a)
                lx0, lx1 = _x_extent(left)
                rx0, rx1 = _x_extent(right)
                left_width, right_width = max(1, lx1 - lx0), max(1, rx1 - rx0)
                gap = rx0 - lx1  # positive: a gap; negative: an overlap
                overlap = max(0, lx1 - rx0)

                if overlap / min(left_width, right_width) >= max_overlap:
                    continue
                # A gap of up to `max_gap`, or a slight overlap of up to 0.3
                # line heights.
                if not (-0.3 * line_height <= gap <= max_gap * line_height):
                    continue
                if abs(left[-1][1] - right[0][1]) >= dy_tol * line_height:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, i, j, left, right)

        if best is None:
            return candidates

        _, i, j, left, right = best
        merged = sorted(left + right, key=lambda p: p[0])
        sources = candidates[i][1] + candidates[j][1]
        candidates = [c for k, c in enumerate(candidates) if k not in (i, j)] + [(merged, sources)]


def _line_height(lines) -> float:
    """Median line height, measured on the bounding polygons."""
    heights = []
    for line in lines:
        if line.boundary:
            ys = [point[1] for point in line.boundary]
            heights.append(max(ys) - min(ys))
    if heights:
        return median(heights)
    # No polygons to measure: estimate the height as 8% of the median
    # baseline length instead.
    return median(baseline_length(line.baseline) for line in lines) * 0.08


def postprocess_segmentation(
    segmentation,
    image,
    min_len_frac: float = 0.12,
    merge_gap: float = 1.5,
    dy_tol: float = 0.6,
    max_overlap: float = 0.3,
    scale: int = 1800,
    repolygonize: bool = True,
    clip_to_region: bool = False,
    restore_reading_order: bool = True,
    topline: bool | None = False,
):
    """
    Merge fragmented baselines, drop spurious short ones, re-polygonize.

    The heuristics assume horizontal lines, so vertical text is passed
    through unchanged.

    Arguments:
        segmentation: A baseline-type segmentation, as returned by
            `KrakenSegmentation`.
        image: The page image the segmentation was computed on. Needed to
            re-derive the line polygons.
        min_len_frac: Drop baselines shorter than this fraction of the
            page's median baseline length. Removes short spurious
            baselines found between the real lines.
        merge_gap: Maximum horizontal gap to bridge when merging fragments,
            in line heights.
        dy_tol: Maximum difference in height where two fragments meet for
            them to be merged, in line heights.
        max_overlap: Maximum horizontal overlap between two fragments, as a
            share of the shorter one, for them to be merged.
        scale: Height in pixels that the page is resized to while the line
            polygons are derived. Lower is faster.
        repolygonize: Whether to re-derive line polygons from the cleaned
            baselines. Turn off to keep the original polygons, which will
            not follow merged baselines.
        clip_to_region: Whether to trim each line polygon to the region it
            belongs to. Fixes the last line of a page running on into blank
            space, but cuts real ink where a region is tighter than its lines.
        restore_reading_order: Whether to re-sort lines afterwards. Merging
            changes the line list, so without this the merged lines end up
            out of reading order.
        topline: Where the segmentation model puts its baselines, as kraken
            defines it for re-polygonization: False for baselines under the
            letters (the usual case), True for hanging baselines along their
            tops, None for centerlines.

    Returns:
        A new `Segmentation`. The input is not modified.
    """
    from kraken.containers import BaselineLine
    from kraken.lib.segmentation import calculate_polygonal_environment

    if segmentation.type != "baselines":
        logger.warning(
            "Segmentation post-processing expects baseline segmentation, got '%s'. Skipping.",
            segmentation.type,
        )
        return segmentation

    if segmentation.text_direction.startswith("vertical"):
        logger.warning(
            "Segmentation post-processing assumes horizontal lines, got '%s'. Skipping.",
            segmentation.text_direction,
        )
        return segmentation

    lines = [line for line in segmentation.lines if line.baseline and len(line.baseline) >= 2]
    if not lines:
        logger.warning("No usable baselines to post-process.")
        return segmentation

    n_before = len(lines)
    line_height = _line_height(lines)
    min_length = min_len_frac * median(baseline_length(line.baseline) for line in lines)

    # --- merge, per region ---------------------------------------------
    # Fragments only ever belong to the same physical line if they are in
    # the same region, so grouping first avoids merging across columns.
    groups: dict[str | None, list[int]] = {}
    for index, line in enumerate(lines):
        key = line.regions[0] if line.regions else None
        groups.setdefault(key, []).append(index)

    merged: list[tuple[list, list[int]]] = []
    for indices in groups.values():
        if len(indices) > 1:
            local = merge_fragments([lines[i].baseline for i in indices], line_height, merge_gap, dy_tol, max_overlap)
            merged.extend((bl, [indices[k] for k in sources]) for bl, sources in local)
        else:
            merged.append((lines[indices[0]].baseline, indices))
    n_merged_away = n_before - len(merged)

    # --- drop short fragments ------------------------------------------
    kept = [(bl, sources) for bl, sources in merged if baseline_length(bl) >= min_length]
    n_dropped = len(merged) - len(kept)
    if not kept:
        logger.warning("Post-processing would drop every line; keeping the original segmentation.")
        return segmentation

    # --- re-polygonize --------------------------------------------------
    baselines = [bl for bl, _ in kept]
    polygons: list[list | None] = [None] * len(kept)
    if repolygonize:
        try:
            polygons = calculate_polygonal_environment(
                image.convert("L"), baselines, scale=(scale, 0), topline=topline
            )
        except Exception as e:
            logger.warning("Re-polygonization failed (%s); keeping original polygons.", e)
            polygons = [None] * len(kept)

    if clip_to_region:
        polygons = _clip_to_regions(polygons, baselines, segmentation)

    # --- rebuild the line records ---------------------------------------
    new_lines = []
    for (baseline, sources), polygon in zip(kept, polygons):
        parent = lines[sources[0]]
        if polygon is None:
            # Either re-polygonization was off or it failed for this line.
            # Reuse the original polygon when the line was not merged, else
            # fall back to a padded box around the baseline.
            polygon = parent.boundary if len(sources) == 1 and parent.boundary else None
        if polygon is None:
            polygon = _box_around(baseline, line_height, image.size)

        new_lines.append(
            BaselineLine(
                id=parent.id if len(sources) == 1 else f"_{uuid.uuid4()}",
                baseline=[(int(x), int(y)) for x, y in baseline],
                boundary=[(int(x), int(y)) for x, y in polygon],
                regions=parent.regions,
                tags=parent.tags,
                base_dir=parent.base_dir,
                imagename=parent.imagename,
            )
        )

    # --- restore reading order ------------------------------------------
    # Merging appends rebuilt lines at the end of the list, so the order
    # kraken established is no longer intact.
    if restore_reading_order and len(new_lines) > 1:
        new_lines = _reorder(new_lines, segmentation)

    logger.info(
        "Segmentation post-processing: %d -> %d lines (merged %d, dropped %d)",
        n_before,
        len(new_lines),
        n_merged_away,
        n_dropped,
    )
    return replace(segmentation, lines=new_lines)


def _clip_to_regions(polygons, baselines, segmentation):
    """
    Trim line polygons to the region they belong to.

    kraken derives a line's polygon from the image alone, so the last line on
    a page, which has no line below it to stop against, tends to run on into
    blank space. Passing the region to kraken's polygonizer as an extra
    boundary (`suppl_obj`) does not prevent this; only cutting the polygon
    down to the region does.

    Off by default, because it cuts real ink whenever a region is drawn
    tighter than its lines.
    """
    import shapely.geometry as geom
    from kraken.lib.segmentation import is_in_region

    regions = [
        geom.Polygon(r.boundary)
        for regs in (segmentation.regions or {}).values()
        for r in regs
        if r.boundary and len(r.boundary) >= 3
    ]
    if not regions:
        return polygons

    clipped = []
    for polygon, baseline in zip(polygons, baselines):
        if polygon is None or len(polygon) < 3:
            clipped.append(polygon)
            continue
        line = geom.LineString(baseline)
        parent = next((r for r in regions if is_in_region(line, r)), None)
        if parent is None:
            clipped.append(polygon)
            continue
        try:
            trimmed = geom.Polygon(polygon).buffer(0).intersection(parent)
        except Exception:
            clipped.append(polygon)
            continue
        if trimmed.is_empty:
            clipped.append(polygon)
            continue
        if trimmed.geom_type == "MultiPolygon":
            trimmed = max(trimmed.geoms, key=lambda g: g.area)
        clipped.append([(int(x), int(y)) for x, y in trimmed.exterior.coords[:-1]])
    return clipped


def _box_around(baseline, line_height: float, size: tuple[int, int]) -> list[tuple[int, int]]:
    """
    A padded rectangle around a baseline, used when polygonization fails.

    Clamped to the image of the given `(width, height)`: near the edge of the
    page the padding would otherwise reach outside it, which kraken refuses to
    extract and PAGE XML cannot express.
    """
    width, height = size
    xs = [point[0] for point in baseline]
    ys = [point[1] for point in baseline]
    pad = int(line_height * 0.6)
    x0, x1 = max(min(xs), 0), min(max(xs), width - 1)
    y0, y1 = max(min(ys) - pad, 0), min(max(ys) + pad, height - 1)
    return [
        (x0, y0),
        (x1, y0),
        (x1, y1),
        (x0, y1),
    ]


def _reorder(lines, segmentation):
    """Re-sort lines into reading order using kraken's own heuristic."""
    from kraken.lib.segmentation import polygonal_reading_order

    text_direction = "rl" if segmentation.text_direction.endswith("-rl") else "lr"
    regions = [r for regs in (segmentation.regions or {}).values() for r in regs]
    try:
        order = polygonal_reading_order(lines, text_direction, regions or None)
        return [lines[i] for i in order]
    except Exception as e:
        logger.warning("Could not restore reading order after post-processing (%s).", e)
        return lines
