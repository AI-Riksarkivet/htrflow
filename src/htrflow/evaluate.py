import os
from collections import Counter
from typing import Any

import jiwer
import pandas as pd
from pagexml.model.physical_document_model import PageXMLPage
from pagexml.parser import parse_pagexml_file
from rich import print
from rich.table import Table
from shapely import GEOSException, Polygon, union_all


class Ratio:
    """
    An unormalized `fraction.Fraction`

    This class makes it easy to compute the total error rate (and other
    fraction-based metrics) of several documents.

    Example: Let A and B be two documents with WER(A) = 1/5 and
    WER(B) = 1/100. In total, there are 2 errors and 105 words, so
    WER(A+B) = 2/105.

    Addition of two `Ratio` instances supports this:
    >>> Ratio(1, 5) + Ratio(1, 100)
    Ratio(2, 105)
    """

    def __init__(self, a, b):
        self.a = int(a)
        self.b = int(b)

    def __add__(self, other):
        if other == 0:
            # sum(a, b) performs the addition 0 + a + b internally,
            # which means that Ratio must support addition with 0 in
            # order to work with sum().
            return self
        return Ratio(self.a + other.a, self.b + other.b)

    __radd__ = __add__  # redirects int + Ratio to __add__

    def __float__(self):
        return -1.0 if self.b == 0 else float(self.a / self.b)

    def __gt__(self, other):
        return float(self) > float(other)

    def __lt__(self, other):
        return float(self) < float(other)

    def __eq__(self, other):
        return float(self) == float(other)

    def __str__(self):
        return f"{self.a}/{self.b}"


class Metric:
    """Metric base class

    All evaluation metrics are implemented by subclassing this class
    and overriding the `compute()` method.

    Example usage:
    >>> metric = CER()
    >>> metric(gt, candidate1)
    {"cer": 0.12}
    >>> metric(gt, candidate2)
    {"cer": 0.09}
    >>> metric.best["cer"](0.12, 0.09)
    0.09

    Attributes:
        best: A dictionary mapping metric labels to a function that
            given a list of evaluation scores returns the best of them.
            Example: {"cer": min}.
    """

    best = {}

    def __call__(self, gt: PageXMLPage, candidate: PageXMLPage) -> dict[str, Any]:
        """Compute a metric

        Arguments:
            gt: Ground truth page
            candidate: Candidate (generated) page

        Returns:
            A dictionary with the name(s) and value(s) of the computed
            metric. For example: {"metricA": 0.5, "metricB": 0.6}
        """
        try:
            return self.compute(self._preprocess(gt), self._preprocess(candidate))
        except (ValueError, GEOSException):
            return {key: None for key in self.best}

    def compute(self, gt: Any, candidate: Any) -> dict[str, Any]:
        pass

    def _preprocess(self, page: PageXMLPage) -> Any:
        pass


class TextMetric(Metric):
    """A metric computed from the pages' text content"""

    def _preprocess(self, page: PageXMLPage) -> str:
        """Extract the text from `page`"""
        lines = []
        for line in page.get_lines():
            text = line.text
            if text is None:
                text = " ".join(word.text for word in line.words)
            lines.append(text)
        return "\n".join(lines)


class RegionMetric(Metric):
    """A metric computed from the pages' region geometries"""

    def _preprocess(self, page: PageXMLPage) -> list[Polygon]:
        """Extract the region polygons from `page`"""
        return [Polygon(region.coords.points) for region in page.get_all_text_regions()]


class LineRegionMetric(Metric):
    """A metric computed from the pages' line geometries"""

    def _preprocess(self, page: PageXMLPage) -> list[Polygon]:
        """Extract the line polygons from `page`"""
        return [Polygon(line.coords.points) for line in page.get_lines()]


class LineCoverage(LineRegionMetric):
    """Compute the line coverage

    Computes how much of the GT lines' combined area is covered by the
    combined area of the predicted lines.
    """

    best = {"line_coverage": max}

    def compute(self, gt: list[Polygon], candidate: list[Polygon]):
        gt = union_all(gt)
        candidate = union_all(candidate)
        return {"line_coverage": Ratio(gt.intersection(candidate).area, gt.area)}


class RegionCoverage(RegionMetric):
    """Compute the region coverage

    Computes how much of the GT regions' combined area is covered by
    the combined area of the predicted regions.
    """

    best = {"region_coverage": max}

    def compute(self, gt: list[Polygon], candidate: list[Polygon]):
        gt = union_all(gt)
        candidate = union_all(candidate)
        return {"region_coverage": Ratio(gt.intersection(candidate).area, gt.area)}


class CER(TextMetric):
    """Compute the character error rate (CER)"""

    best = {"cer": min}

    def compute(self, gt: str, candidate: str):
        cer = jiwer.process_characters(gt, candidate)
        errors = cer.insertions + cer.deletions + cer.substitutions
        return {"cer": Ratio(errors, errors + cer.hits)}


class WER(TextMetric):
    """Compute the word error rate (WER)"""

    best = {"wer": min}

    def compute(self, gt: str, candidate: str):
        wer = jiwer.process_words(gt, candidate)
        errors = wer.insertions + wer.deletions + wer.substitutions
        return {"wer": Ratio(errors, errors + wer.hits)}


class BagOfWords(TextMetric):
    """Compute the intersection of the pages' bags-of-words

    This metric is reading order agnostic.
    """

    best = {"bow_hits": max, "bow_extras": min}

    def compute(self, gt: str, candidate: str):
        gt_bow = Counter(gt.split())
        candidate_bow = Counter(candidate.split())

        gt_words = sum(gt_bow.values())
        candidate_words = sum(candidate_bow.values())
        intersection_words = sum((gt_bow & candidate_bow).values())

        return {
            "bow_hits": Ratio(intersection_words, gt_words),
            "bow_extras": Ratio(candidate_words - intersection_words, candidate_words),
        }


def _df_to_table(df: pd.DataFrame, highlight_policy: dict) -> Table:
    """Convert a pandas dataframe to a rich table

    Highlights the best entry of each row according to the given policy.

    Arguments:
        df: Input dataframe
        highlight_policy: A dictionary mapping indices of `df` to a
            function that given a list of values returns the value that
            should be highlighted in the table.
    """

    table = Table()
    table.add_column()
    for column in df.columns:
        table.add_column(str(column))
    for label, row in zip(df.index, df.values):
        best = highlight_policy.get(label, lambda _: None)(row)
        table.add_row(label, *[_format_value(value, best) for value in row])
    return table


def _format_value(value: float, best: float | None) -> str:
    """Returns a formatted string representing `value`

    Arguments:
        value: The value to be formatted
        best: If value == best, the formatted string will be highlighted
            bold green.
    """
    res = f"{100 * value:.4}" if value is not None else "None"
    if value == best:
        res = "[bold green]" + res + "[/bold green]*"
    return res


def print_summary(results: pd.DataFrame, metrics: list[Metric]):
    """Pretty-print a summary of `results`"""
    highlight_policy = {}
    for metric in metrics:
        highlight_policy |= metric.best
    summary = results.sum().map(float).unstack("pipeline")
    table = _df_to_table(summary, highlight_policy)
    print(table)


def read_xmls(directory: str) -> dict[str, PageXMLPage]:
    """Read PageXMLs from a directory

    Returns a dictionary mapping the PageXML file name to its
    corresponding `PageXMLPage` instance.
    """
    pages = {}
    for parent, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(".xml"):
                continue
            try:
                page = parse_pagexml_file(os.path.join(parent, file))
            except ValueError:
                continue
            pages[file] = page
    return pages


def evaluate(gt_directory: str, *candidate_directories: tuple[str, ...]) -> pd.DataFrame:
    """Evaluate candidate(s) against GT

    The candidates must have the same file name as their corresponding
    GT files. For example, `candidate_directory/page_0.xml` is evaluated
    against `gt_directory/page_0.xml`.

    Pretty-prints a summary of the results and returns all results as a
    pandas DataFrame.

    Arguments:
        gt: Path to directory with GT PageXMLs
        candidate_directories: Paths to directories with generated PageXMLs

    Returns:
        A pandas DataFrame with evaluation results.
    """
    metrics = [CER(), WER(), BagOfWords(), LineCoverage(), RegionCoverage()]
    gt = read_xmls(gt_directory)

    dfs = []
    for candidate in candidate_directories:
        candidate_pages = read_xmls(candidate)
        pages = [page for page in gt if page in candidate_pages and gt[page].num_words > 0]
        for metric in metrics:
            values = [metric(gt[page], candidate_pages[page]) for page in pages]
            df = pd.DataFrame(values, index=pages)
            dfs.append(df)

    df = pd.concat(dfs, axis=1)
    candidates = [os.path.basename(path) for path in candidate_directories]
    metric_labels = df.columns[: len(df.columns) // len(candidates)]
    names = ["pipeline", "metric"]
    df.columns = pd.MultiIndex.from_product([candidates, metric_labels], names=names)
    print_summary(df, metrics)
    return df
