import logging
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy2
from typing import NotRequired, TypedDict

import mlflow
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from samisk_ocr.trocr.config import Config
from samisk_ocr.utils import setup_logging

logger = logging.getLogger(__name__)


def get_best_checkpoint_from_mlflow_final_evaluation(run_name: str, experiment_name: str) -> str:
    config = Config()
    mlflow.set_tracking_uri(config.mlflow_url)
    mlflow.set_experiment(experiment_name)
    print(f"run_name: {run_name}")
    print(f"experiment_name: {experiment_name}")
    # Specify what model we want to load
    run_id = mlflow.search_runs(filter_string=f"run_name = '{run_name}'")["run_id"].item()  # type: ignore

    return mlflow.artifacts.load_dict(f"runs:/{run_id}/final_evaluation/min_cer.json")[
        "min_cer_checkpoint"
    ]


def upload_model(
    model_dir: Path,
    model_name: str,
    private: bool,
    checkpoint: int | None,
    experiment_name: str | None,
):
    """
    Upload a TrOCR model to huggingface
    
    Arguments:
    ----------
    model_dir : Path
        Directory where TrOCR model checkpoints and processor are stored
    model_name : str
        Name the model should have on huggingface
    private : bool
        If flagged, will upload model as private
    checkpoint : int | None
        Training checkpoint iteration to use (optional). Example: 43132
        If not specified, it will use the best modell from the final_evaluation. 
        (This assumes that final evaluation has been run)
    experiment_name : str | None
        The name of the experiment to evaluate, e.g. 'TrOCR trocr-base-printed finetuning'
        If checkpoint is not specified, this is required. If checkpoint is specified, this is ignored.
    """
    model_p = model_dir / "final_model"
    processor_p = model_dir / "processor"

    if checkpoint is None:
        if experiment_name is None:
            raise ValueError(
                "If checkpoint is not specified, experiment_name must be specified"
            )
        checkpoint_name = get_best_checkpoint_from_mlflow_final_evaluation(
            run_name=model_dir.name, experiment_name=experiment_name
        )
        model_p = model_dir / checkpoint_name
    else:
        model_p = model_dir / f"checkpoint-{checkpoint}"

    device = torch.device("cpu")

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


if __name__ == "__main__":
    parser = ArgumentParser(description="Upload a TrOCR model to huggingface")
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory where TrOCR model checkpoints and processor are stored",
    )
    parser.add_argument("model_name", help="Name the model should have on huggingface")
    parser.add_argument(
        "--checkpoint",
        type=int,
        help="Training checkpoint iteration to use (will use final_model if not specified). Example: 43132",
        default=None,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="The name of the experiment to evaluate, e.g. 'TrOCR trocr-base-printed finetuning'",
        default=None,
    )
    parser.add_argument(
        "--public", help="If flagged, will upload model as public", action="store_true"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    args = parser.parse_args()

    setup_logging(source_script="model_to_hf_hub", log_level=args.log_level)
    logger.info(args)

    upload_model(
        model_dir=args.model_dir,
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        private=not args.public,
        experiment_name=args.experiment_name,
    )
