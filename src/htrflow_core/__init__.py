from collections import Counter


try:
    from importlib.metadata import PackageNotFoundError, metadata, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, metadata, version


def _package_metadata_as_dict(package_name: str) -> dict:
    """Get package metadata and return it as a dict."""
    m = metadata(package_name)

    exclude_keys = []
    result = {}

    for key, count in Counter(m.keys()).items():
        if key in exclude_keys:
            continue

        if count == 1:
            result[key] = m[key]
            continue

        result[key] = m.get_all(key)

    if "Project-URL" in result:
        urls = result["Project-URL"]
        result["Project-URL"] = {}
        for entry in urls:
            key, val = entry.split(", ", 1)  #
            result["Project-URL"][key.strip()] = val.strip()

    return result


try:
    __package_name__ = __name__ if __name__ else "htrflow"
    meta = _package_metadata_as_dict(__package_name__)
    __version__ = meta.get("Version", "unknown")
    __author__ = meta.get("Author", "unknown")
    __desc__ = meta.get("Summary", "unknown")
    __name__ = meta.get("Name", __package_name__)
except PackageNotFoundError:
    __version__ = "unknown"
