from collections import Counter


try:
    from importlib.metadata import PackageNotFoundError, metadata, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, metadata, version


def _package_metadata_as_dict(package_name: str, exclude_keys=None) -> dict:
    """Get package metadata and return it as a dict, excluding specified keys."""
    if exclude_keys is None:
        exclude_keys = []

    try:
        meta_data = metadata(package_name)
    except PackageNotFoundError:
        return {"Version": "unknown", "Author": "unknown", "Summary": "unknown", "Name": package_name}

    result = {}
    keys_counter = Counter(meta_data.keys())

    for key in keys_counter:
        if key in exclude_keys:
            continue

        values = meta_data.get_all(key) if keys_counter[key] > 1 else meta_data[key]
        if key == "Project-URL" and keys_counter[key] > 1:
            result[key] = {k.strip(): v.strip() for k, v in (entry.split(", ", 1) for entry in values)}
        else:
            result[key] = values

    return result


__package_name__ = __name__
meta = _package_metadata_as_dict(__package_name__, exclude_keys=None)
__version__ = meta.get("Version", "unknown")
__author__ = meta.get("Author", "unknown")
__desc__ = meta.get("Summary", "unknown")
__name__ = meta.get("Name", __package_name__)
