from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
from utils import setup_logging
from jiwer import wer, cer
import logging
import json

logger = logging.getLogger(__name__)


def evaluate_collection_level(df: pd.DataFrame) -> tuple[float, float]:
    """Calculate WER and CER for the entire collection"""
    coll_wer = wer(
        reference=df.ground_truth.to_list(), hypothesis=df.transcription.to_list()
    )
    coll_cer = cer(
        reference=df.ground_truth.to_list(), hypothesis=df.transcription.to_list()
    )
    return (coll_wer, coll_cer)


def evaluate_page_level(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate WER and CER for the each page"""
    df["wer"] = df.apply(
        lambda row: wer(reference=row.ground_truth, hypothesis=row.transcription),
        axis=1,
    )
    df["cer"] = df.apply(
        lambda row: cer(reference=row.ground_truth, hypothesis=row.transcription),
        axis=1,
    )
    return df


def evaluate_line_level(
    ground_truth: str, predicted_transcription: str
) -> pd.DataFrame:
    """Calculate WER and CER for each line in the texts"""
    ground_truth_lines = [line for line in ground_truth.split("\n") if line]
    prediction_lines = [line for line in predicted_transcription.split("\n") if line]

    if len(ground_truth_lines) < len(prediction_lines):
        ground_truth_lines += [""] * (len(prediction_lines) - len(ground_truth_lines))
    if len(prediction_lines) < len(ground_truth_lines):
        prediction_lines += [""] * (len(ground_truth_lines) - len(prediction_lines))

    wers = [
        wer(reference=trans, hypothesis=pred)
        for trans, pred in zip(ground_truth_lines, prediction_lines)
    ]
    cers = [
        cer(reference=trans, hypothesis=pred)
        for trans, pred in zip(ground_truth_lines, prediction_lines)
    ]
    return pd.DataFrame(
        {
            "ground_truth": ground_truth_lines,
            "transcription": prediction_lines,
            "wer": wers,
            "cer": cers,
        }
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate transcriptions")
    parser.add_argument(
        "predictions",
        type=Path,
        help=".csv file with predicted transcriptions",
    )
    parser.add_argument(
        "transcriptions",
        type=Path,
        help="The directory containing ground truth transcriptions (page-level .txt-files)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="The output directory to store evaluation results",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    args = parser.parse_args()
    setup_logging(source_script="evaluate_predictions", log_level=args.log_level)

    df = pd.read_csv(args.predictions)

    ground_truth_paths = df.image.apply(Path).apply(
        lambda p: args.transcriptions / (p.stem + ".txt")
    )
    if not all(ground_truth_paths.apply(lambda p: p.exists())):
        logger.error(
            f"Some transcriptions for images in {args.predictions} do not exist in {args.transcriptions}"
        )
        exit()

    df["ground_truth"] = ground_truth_paths.apply(lambda p: p.read_text())

    args.output_dir.mkdir(parents=True)

    # Calculate WER and CER for the entire collection
    coll_wer, coll_cer = evaluate_collection_level(df)
    with (args.output_dir / "all_pages.json").open("w+") as f:
        f.write(json.dumps({"CER": coll_cer, "WER": coll_wer}))

    # Calculate WER and CER for each oage
    df = evaluate_page_level(df)
    df.to_csv(args.output_dir / "page_level.csv", index=False)

    # Calculate WER and CER for each line in each text
    (args.output_dir / "line_level").mkdir()
    for e in df.itertuples():
        df = evaluate_line_level(
            ground_truth=e.ground_truth, predicted_transcription=e.transcription
        )
        df.to_csv(args.output_dir / "line_level" / f"{e.image}.csv", index=False)

    logger.info(f"See WER and CER scores in {args.output_dir}")
