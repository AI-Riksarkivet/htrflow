from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from htrflow.volume import volume


def average_text_confidence(node: "volume.ImageNode") -> float:
    """Average text confidence value of `node` and its children

    Returns the average text confidence of all text lines attached
    to `node` (at any depth). The input node may be a page or a region.

    Arguments:
        node: Input node.

    Returns:
        The node's average text confidence score, or 0.0 if it couldn't be
        computed (e.g., if the node didn't contain any text lines).
    """
    nodes = node.traverse(lambda node: node.is_line())
    if nodes:
        return sum(line_text_confidence(node) for node in nodes) / len(nodes)
    return 0.0


def line_text_confidence(node: "volume.SegmentNode") -> float:
    """The text confidence score of `node`"""
    if text_result := node.text_result:
        return text_result.top_score()
    return 0.0
