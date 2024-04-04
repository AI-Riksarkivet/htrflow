from copy import deepcopy

from htrflow_core.utils.geometry import estimate_printspace, get_region_location
from htrflow_core.volume import volume


def remove_noise_regions(volume: volume.Volume, threshold: float = 0.8):
    """Remove noise regions from volume

    Makes a copy of the given volume where noisy regions are removed.
    Uses the heuristic defined in `is_noise`.

    Arguments:
        volume: Input volume with text and regions
        threshold: The confidence score threshold, default 0.8.

    Returns:
        A copy of `volume` where all regions have an average text
        recognition confidence score above the given threshold.
    """
    volume = deepcopy(volume)
    volume.prune(lambda node: is_noise(node, threshold), include_starting_node=False)
    return volume


def is_noise(node: volume.BaseDocumentNode, threshold: float = 0.8):
    """Heuristically determine if region is noise

    Assumes that a region is noise if the average text recognition
    confidence score is lower than the given threshold.

    Arguments:
        node: Which node to check
        threshold: Threshold for the average text recognition
            confidence score. Defaults to 0.8, i.e., any region with
            avg. confidence lower than 0.8 is regarded as noise.

    Returns:
        True if `node` is a region (i.e. parent to nodes with text lines)
        and the average text recognition confidence score of its
        children is below `threshold`.
    """
    if node.children and all(child.text for child in node):
        conf = sum(child.get("text_result").top_score() for child in node) / len(node.children)
        return conf < threshold
    return False


def label_regions(volume: volume.Volume, key="region_location"):
    """Label volume's regions

    Labels each top-level segment of the volume as one of the five
    region types specified by geometry.RegionLocation. Saves the label
    in the node's data dictionary under `key`.

    Arguments:
        volume: Input volume
        key: Key used to save the region label. Defaults to
            "region_location".
    """

    for page in volume:
        printspace = estimate_printspace(page.image)
        for node in page:
            node.add_data(**{key: get_region_location(printspace, node.bbox)})
