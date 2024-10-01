import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy2
from typing import TypedDict

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from samisk_ocr.utils import setup_logging

logger = logging.getLogger(__name__)


def upload_model(base_dir: Path, run_name: str, model_name: str, private: bool):
    model_p = base_dir / run_name / "final_model"
    processor_p = base_dir / run_name / "processor"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # check that model loads
    processor = TrOCRProcessor.from_pretrained(processor_p)
    model = VisionEncoderDecoderModel.from_pretrained(model_p).to(device)

    out_model_p = Path(f"trocr_models/{model_name}")
    out_model_p.mkdir(parents=True, exist_ok=True)
    for e in model_p.iterdir():
        copy2(src=e, dst=out_model_p / e.name)

    for e in processor_p.iterdir():
        copy2(src=e, dst=out_model_p / e.name)

    # check that model loads
    processor = TrOCRProcessor.from_pretrained(out_model_p)
    model = VisionEncoderDecoderModel.from_pretrained(out_model_p).to(device)

    # push to hub
    repo_id = f"Sprakbanken/{model_name}"
    model.push_to_hub(repo_id=repo_id, private=private)
    processor.push_to_hub(repo_id=repo_id, private=private)

    # check that model loads
    processor = TrOCRProcessor.from_pretrained(repo_id)
    model = VisionEncoderDecoderModel.from_pretrained(repo_id).to(device)

    logger.info("Pushed model to https://huggingface.co/%s", repo_id)


class ModelToUpload(TypedDict):
    run_name: str
    model_name: str


def parse_model_runs_and_names(runs_and_names: list[str]) -> list[ModelToUpload]:
    run_names = [e[len("run_name=") :] for e in runs_and_names if e.startswith("run_name=")]
    model_names = [e[len("model_name=") :] for e in runs_and_names if e.startswith("model_name=")]
    if not (
        len(run_names) + len(model_names) == len(runs_and_names)
        and len(run_names) == len(model_names)
    ):
        raise Exception(
            "Malformatted model runs and names. Should be one or more pairs of 'run_name=<run_name> model_name<model_name>'"
        )

    return [
        {"run_name": run_name, "model_name": model_name}
        for run_name, model_name in zip(run_names, model_names)
    ]


if __name__ == "__main__":
    parser = ArgumentParser(description="Upload TrOCR models to huggingface")
    parser.add_argument("base_dir", type=Path, help="Base dir where TrOCR models are stored")
    parser.add_argument(
        "model_runs_and_names",
        nargs="+",
        help="Pairs of run name and model name for models to upload. Example: run_name=sweet-pig-123 model_name=smi_best",
    )
    parser.add_argument(
        "--public", help="If flagged, will upload models as public", action="store_true"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    args = parser.parse_args()
    models_to_upload = parse_model_runs_and_names(args.model_runs_and_names)

    setup_logging(source_script="models_to_hf_hub", log_level=args.log_level)
    logger.info(vars(args))

    for e in models_to_upload:
        upload_model(base_dir=args.base_dir, **e, private=not args.public)

    logger.info("Finished uploading models.")
