from htrflow.document import Region


def average_text_confidence(region: "Region") -> float:
    """
    Average text confidence value of `region` and its children

    Returns the average text confidence of all texts attached to
    `region`, at any depth.

    Arguments:
        region: Input region.

    Returns:
        The region's average text confidence score, or 0.0 if it couldn't be
        computed (e.g., if the region didn't contain any text lines).
    """
    regions = list(region.traverse())
    if regions:
        return sum(text_confidence(region) for region in regions) / len(regions)
    return 0.0


def text_confidence(region: "Region") -> float:
    """The text confidence score of `region`"""
    if region.transcription:
        return max(region.transcription, key=lambda t: t.confidence or 0.0).confidence
    return 0.0
