from copy import deepcopy

from htrflow_core.volume import volume


def remove_noise_regions(page: volume.PageNode, threshold: float = 0.8):
    """Remove noise regions from page

    Makes a copy of the given volume where noisy regions are removed.
    Uses the heuristic defined in `is_noise`.

    Arguments:
        page: Input page with text and regions
        threshold: The confidence score threshold, default 0.8.

    Returns:
        A copy of `page` where all regions have an average text
        recognition confidence score above the given threshold.
    """
    page = deepcopy(page)
    page.prune(lambda node: is_noise(node, threshold), include_starting_node=False)
    return page


def is_noise(node: volume.ImageNode, threshold: float = 0.8):
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
