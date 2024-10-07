from collections import Counter
from importlib.metadata import PackageNotFoundError, metadata
from typing import Counter, Dict, List, Optional

import htrflow


def _package_metadata_as_dict(package_name: str, exclude_keys: Optional[List[str]] = None) -> Dict:
    """Get package metadata and return it as a dict, excluding specified keys."""
    if exclude_keys is None:
        exclude_keys = []
    try:
        meta_data = metadata(package_name)

    except PackageNotFoundError:
        return {
            "Version": "unknown",
            "Author": "unknown",
            "Summary": "unknown",
            "Name": package_name,
        }

    filtered_metadata = {}
    keys_counter: Counter[str] = Counter(meta_data.keys())

    for key in keys_counter:
        if key in exclude_keys:
            continue

        values = meta_data.get_all(key) if keys_counter[key] > 1 else meta_data[key]

        if key == "Project-URL" and keys_counter[key] > 1:
            filtered_metadata[key] = {k.strip(): v.strip() for k, v in (entry.split(", ", 1) for entry in values)}
        else:
            filtered_metadata[key] = values

    return filtered_metadata


meta = _package_metadata_as_dict(
    htrflow.__package__,
    exclude_keys=[
        "Classifier",
        "Readme",
        "Description",
        "Description-Content-Type",
        "License",
    ],
)
