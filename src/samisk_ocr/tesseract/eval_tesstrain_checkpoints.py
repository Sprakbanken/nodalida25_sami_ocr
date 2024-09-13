"""Motivert av https://github.com/tesseract-ocr/tesseract/issues/3560"""

import concurrent.futures
import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy2

import pandas as pd

from samisk_ocr.metrics import compute_cer, compute_wer
from samisk_ocr.tesseract.transcribe import transcribe
from samisk_ocr.utils import setup_logging

logger = logging.getLogger(__name__)


def copy_models_to_tessdata(model_dir: Path, tessdata_dir: Path):
    for model in model_dir.iterdir():
        copy2(model, dst=tessdata_dir / model.name)


def remove_models_from_tessdata(model_dir: Path, tessdata_dir: Path):
    for model in model_dir.iterdir():
        tessdata_model = tessdata_dir / model.name
        if tessdata_model.exists():
            tessdata_model.unlink()


def transcribe_w_model(model_name: str, image_dir: Path, gt_df: pd.DataFrame, output_file: Path):
    if output_file.exists():
        return
    try:
        transcription_df = transcribe(model_name=model_name, image_dir=image_dir, line_level=True)
    except Exception as ex:
        logger.warning(
            f"Encountered exception when running model {model_name} on image dir {image_dir.name}"
        )
        return
    df = transcription_df.merge(gt_df, on="image")
    WER_concat = compute_wer(
        transcription=" ".join(df.transcription), ground_truth=" ".join(df.ground_truth)
    )
    CER_concat = compute_cer(
        transcription=" ".join(df.transcription), ground_truth=" ".join(df.ground_truth)
    )
    scores = {
        "model_name": model_name,
        "iteration": int(model_name.split("_")[-1]),
        "CER": CER_concat,
        "WER": WER_concat,
    }
    with output_file.open("w+") as f:
        json.dump(scores, f, indent=4)


def call_transcribe_with_params(params):
    transcribe_w_model(*params)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate tesseract model checkpoints")
    parser.add_argument("model_names", help="Names of tesseract models", nargs="+")
    parser.add_argument(
        "--tessdata", help="Path to tessdata (see tesseract_howto)", type=Path, required=True
    )
    parser.add_argument(
        "--tesstrain_repo", type=Path, help="Path to tesstrain repo", default=Path("tesstrain")
    )
    parser.add_argument(
        "--dataset_path", type=Path, help="Path to dataset", default=Path("data/samisk_ocr_dataset")
    )
    parser.add_argument("--splits", help="dataset splits", default=["val"])
    parser.add_argument(
        "--output_dir", type=Path, help="Path to directory to store plots", required=True
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    setup_logging("eval_tesstrain_checkpoints", log_level="DEBUG")

    all_params = []

    for model_name in args.model_names:
        checkpoint_models_dir = args.tesstrain_repo / "data" / model_name / "tessdata_best/"
        assert checkpoint_models_dir.exists()

        copy_models_to_tessdata(checkpoint_models_dir, args.tessdata)

        for split in args.splits:
            gt_df = pd.read_csv(args.dataset_path / split / "metadata.csv")
            gt_df["image"] = gt_df.file_name.apply(lambda x: Path(x).name)
            gt_df = gt_df.rename(columns={"text": "ground_truth"})

            image_dir = args.dataset_path / split
            output_dir = args.output_dir / split
            output_dir.mkdir(exist_ok=True, parents=True)

            for checkpoint_model in checkpoint_models_dir.iterdir():
                checkpoint_model_name = checkpoint_model.name[: -len(".traineddata")]
                output_file = output_dir / f"{checkpoint_model_name}.json"
                all_params.append((checkpoint_model_name, image_dir, gt_df, output_file))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(call_transcribe_with_params, all_params)

    # Remove checkpoint models from tessdata
    for model_name in args.model_names:
        checkpoint_models_dir = args.tesstrain_repo / "data" / model_name / "tessdata_best/"
        remove_models_from_tessdata(checkpoint_models_dir, args.tessdata)
