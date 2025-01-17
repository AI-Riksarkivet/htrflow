import logging
import multiprocessing
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from uuid import uuid4

import cv2
import numpy as np
import pydantic
from huggingface_hub import model_info, snapshot_download
from laia.common.arguments import CommonArgs, DataArgs, DecodeArgs, TrainerArgs
from laia.scripts.htr.decode_ctc import run as decode

from htrflow.models.base_model import BaseModel
from htrflow.results import Result
from htrflow.utils.imgproc import pad_image


logger = logging.getLogger(__name__)


class PyLaia(BaseModel):
    """
    A minimal HTRflow-style model wrapper around PyLaia.

    Uses Teklia's implementation of PyLaia. For further
    information, see:
    https://atr.pages.teklia.com/pylaia/usage/prediction/#decode-arguments

    Example usage with the `TextRecognition` step:
    ```yaml
    - step: TextRecognition
      settings:
        model: PyLaia
        model_settings:
          model: Teklia/pylaia-belfort
          device: cuda
          revision: d35f921605314afc7324310081bee55a805a0b9f
        generation_settings:
          batch_size: 8
          temperature: 1
    ```
    """

    IMAGE_ID_PATTERN = r"(?P<image_id>[-a-z0-9]{36})"
    CONFIDENCE_PATTERN = r"(?P<confidence>[0-9.]+)"
    TEXT_PATTERN = r"\s*(?P<text>.*)\s*"
    LINE_PREDICTION = re.compile(rf"{IMAGE_ID_PATTERN} {CONFIDENCE_PATTERN} {TEXT_PATTERN}")

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        **kwargs,
    ):
        """
        Arguments:
            model (str):
                The Hugging Face Hub repository ID or a local path with PyLaia artifacts:
                - weights.ckpt
                - syms.txt
                - (optionally) language_model.arpa.gz, lexicon.txt, tokens.txt
            revision: Optional revision of the Huggingface repository.
            kwargs:
                Additional kwargs passed to BaseModel.__init__ (e.g., 'device').
        """
        super().__init__(**kwargs)

        model_info_dict = get_pylaia_model(model, revision=revision)
        self.model_dir = model_info_dict.model_dir
        model_version = model_info_dict.model_version
        self.use_language_model = model_info_dict.use_language_model
        self.language_model_params = model_info_dict.language_model_params

        self.metadata.update(
            {
                "model": model,
                "model_version": model_version,
            }
        )

        logger.info(f"Initialized PyLaiaModel from '{model}' on device '{self.device}'.")

    def _predict(self, images: list[np.ndarray], **decode_kwargs) -> list[Result]:
        """
        PyLaia-specific prediction method: runs text recognition.

        Args:
            images (list[np.ndarray]):
                List of images as NumPy arrays (e.g., shape [H, W, C]).
            batch_size (int, optional):
                Batch size for decoding. Defaults to 1.
            reading_order (str, optional):
                Reading order for text recognition. Defaults to "LTR".
            num_workers (int, optional):
                Number of workers for parallel processing. Defaults to `multiprocessing.cpu_count()`.

        Returns:
            list[Result]:
                A list of Result objects containing recognized text and
                optionally confidence scores.
        """

        temperature = decode_kwargs.get("temperature", 1.0)
        batch_size = decode_kwargs.get("batch_size", 1)
        reading_order = decode_kwargs.get("reading_order", "LTR")
        num_workers = decode_kwargs.get("num_workers", multiprocessing.cpu_count())

        common_args = CommonArgs(
            checkpoint="weights.ckpt",
            train_path=str(self.model_dir),
            experiment_dirname="",
        )

        data_args = DataArgs(
            batch_size=batch_size, color_mode="L", reading_order=reading_order, num_workers=num_workers
        )

        gpus_flag = 1 if self.device.type == "cuda" else 0
        trainer_args = TrainerArgs(gpus=gpus_flag)

        decode_args = DecodeArgs(
            include_img_ids=True,
            join_string="",
            convert_spaces=True,
            print_line_confidence_scores=True,
            print_word_confidence_scores=False,
            temperature=temperature,
            use_language_model=self.use_language_model,
            **self.language_model_params.model_dump(),
        )

        # Note: PyLaia's 'decode' function expects disk-based file paths rather than in-memory data.
        # Because it is tightly integrated as a CLI tool, we must create temporary image files
        # and pass their paths to the PyLaia decoder. Otherwise, PyLaia cannot process these images.
        tmp_images_dir = Path(mkdtemp())
        logger.debug(f"Created temp folder for images: {tmp_images_dir}")

        image_ids = [str(uuid4()) for _ in images]

        for img_id, np_img in zip(image_ids, images):
            padded_img = _ensure_min_width(np_img, 120, 125)  # Just to fix the min pixel width issue
            cv2.imwrite(str(tmp_images_dir / f"{img_id}.jpg"), padded_img)

        with NamedTemporaryFile() as pred_stdout, NamedTemporaryFile() as img_list:
            Path(img_list.name).write_text("\n".join(image_ids))

            with redirect_stdout(open(pred_stdout.name, mode="w")):
                decode(
                    syms=str(self.model_dir / "syms.txt"),
                    img_list=img_list.name,
                    img_dirs=[str(tmp_images_dir)],
                    common=common_args,
                    data=data_args,
                    trainer=trainer_args,
                    decode=decode_args,
                    num_workers=num_workers,
                )
                sys.stdout.flush()

            decode_output_lines = Path(pred_stdout.name).read_text().strip().splitlines()

        results = []
        metadata = self.metadata | {"decode_kwargs": decode_kwargs}

        for line in decode_output_lines:
            match = self.LINE_PREDICTION.match(line)
            if not match:
                logger.warning("Could not parse line: %s", line)
                continue
            _, score_str, text = match.groups()  # _ = image_id

            try:
                score_val = float(score_str)
            except ValueError:
                score_val = 0.0

            result = Result.text_recognition_result(metadata, [text], [score_val])
            results.append(result)

        logger.debug(f"PyLaia recognized {len(results)} lines of text.")

        return results


class LanguageModelParams(pydantic.BaseModel):
    """Pydantic model for language model parameters."""

    language_model_weight: float = 1.0
    language_model_path: str = ""
    lexicon_path: str = ""
    tokens_path: str = ""



class PyLaiaModelInfo(pydantic.BaseModel):
    """
    Pydantic model specifying what `get_pylaia_model` should return.
    """

    model_config = pydantic.ConfigDict(protected_namespaces=())

    model_dir: Path
    model_version: str
    use_language_model: bool
    language_model_params: LanguageModelParams


def get_pylaia_model(
    model: str,
    revision: str | None = None,
    cache_dir: str | None = ".cache",
) -> PyLaiaModelInfo:
    """
    Encapsulates logic for retrieving a PyLaia model (from either a local path
    or by downloading from the Hugging Face Hub), and detecting whether a
    language model is available.

    Args:
        model (str):
            - If this is a valid local directory path, we assume it contains
              the necessary PyLaia files and use that directly.
            - Otherwise, we treat `model` as a Hugging Face Hub repo_id and
              download from the HF Hub.
        revision (str | None, optional):
            Git branch, tag, or commit SHA to download from the HF Hub.
            If None, the default branch or tag is used.
        cache_dir (str | None, optional):
            Path to the folder where cached files are stored. Defaults to ".cache".

    Returns:
        PyLaiaModelInfo: A data class with these fields:
            - model_dir (Path): Local path to the model directory
            - model_version (str): "local" if loaded from directory, or commit SHA if from HF
            - use_language_model (bool): Whether a language model is present
            - language_model_params (dict[str, Any]): Additional arguments for LM
    """

    model_dir, model_version = _download_or_local_path(model, revision, cache_dir)
    use_language_model, language_model_params = _detect_language_model(model_dir)

    logger.debug(f"Model directory: {model_dir}")
    logger.debug(f"Model version: {model_version}")
    logger.debug(f"Use language model: {use_language_model}")

    return PyLaiaModelInfo(
        model_dir=model_dir,
        model_version=model_version,
        use_language_model=use_language_model,
        language_model_params=language_model_params,
    )


def _download_or_local_path(
    model: str,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> tuple[Path, str]:
    """
    If 'model' is a local directory, model_version = "local".
    Otherwise, fetch from HF, and model_version = commit SHA (optionally
    at a specific `revision`).
    """
    model_path = Path(model)
    if model_path.is_dir():
        logger.info(f"Using local PyLaia model from: {model_path}")
        return model_path, "local"
    else:
        logger.info(f"Downloading/loading PyLaia model '{model}' from the Hugging Face Hub...")
        downloaded_dir = Path(snapshot_download(repo_id=model, revision=revision, cache_dir=cache_dir))
        if revision:
            version_sha = model_info(model, revision=revision).sha
        else:
            version_sha = model_info(model).sha

        return downloaded_dir, version_sha


def _detect_language_model(model_dir: Path) -> tuple[bool, LanguageModelParams]:
    """
    Checks if 'tokens.txt' is present in the model_dir, and if so,
    updates language model parameters accordingly.
    """
    tokens_file = model_dir / "tokens.txt"
    use_language_model = tokens_file.exists()
    language_model_params = {"language_model_weight": 1.0}

    if use_language_model:
        arpa_file = model_dir / "language_model.arpa.gz"
        lexicon_file = model_dir / "lexicon.txt"

        language_model_params.update(
            {
                "language_model_path": str(arpa_file) if arpa_file.exists() else "",
                "lexicon_path": str(lexicon_file) if lexicon_file.exists() else "",
                "tokens_path": str(tokens_file),
            }
        )

    return use_language_model, language_model_params


def _ensure_min_width(img: np.ndarray, min_width: int, target_width: int) -> np.ndarray:
    """
    Ensures an image meets a minimum width by padding if necessary.

    Args:
        img (np.ndarray): Input image.
        min_width (int): Minimum width before padding.
        target_width (int): Final width after padding.

    Returns:
        np.ndarray: The padded image (if needed).
    """
    _, width = img.shape[:2]
    if width < min_width:
        total_pad = target_width - width
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad
        return pad_image(img, pad_left=left_pad, pad_right=right_pad)
    return img
