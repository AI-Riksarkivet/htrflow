import os
import sys
import warnings


# Fix warnings in for openmmlab..
class SuppressOutput:
    def __init__(self, show_mmengine_warnings: bool = True) -> None:
        self.show_mmengine_warnings = show_mmengine_warnings

    def __enter__(self):
        if self.show_mmengine_warnings:
            warnings.filterwarnings("ignore")
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.show_mmengine_warnings:
            sys.stdout.close()
            sys.stdout = self._original_stdout
            sys.stderr.close()
            sys.stderr = self._original_stderr
