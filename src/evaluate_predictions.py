from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
from utils import setup_logging
from jiwer import wer, cer
import logging
import json
from functools import partial
from map_transkribus_lines_to_gt_lines import (
    map_transkribus_image_lines_to_gt_image_lines,
)

logger = logging.getLogger(__name__)


def urn_to_langcode(urn: str) -> str:
    df = pd.read_csv("data/testset_languages.tsv", sep="\t")

    # expect full match on page level
    if urn in df.side_filnavn:
        return df[df.side_filnavn == urn].språkkode.item()

    # line level images contain line number and bbox information in filename
    for e in df.side_filnavn:
        if e in urn:
            return df[df.side_filnavn == e].språkkode.item()
    return None


def evaluate_collection_level(df: pd.DataFrame) -> tuple[float, float]:
    """Calculate WER and CER across rows in df"""
    coll_wer = wer(
        reference=df.ground_truth.to_list(), hypothesis=df.transcription.to_list()
    )
    coll_cer = cer(
        reference=df.ground_truth.to_list(), hypothesis=df.transcription.to_list()
    )
    return (coll_wer, coll_cer)


def evaluate_each_row(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate WER and CER for the each row in df"""
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
    """Calculate WER and CER for each line in the texts (assumes text is page level/multiple lines)"""
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


def find_gt(img_path: Path, gt_dir: Path) -> Path:
    exact_match = next(gt_dir.glob(f"{img_path.stem}*.txt"), None)
    if exact_match:
        return exact_match
    logger.error(f"Couldn't find transcription for image {img_path} in {gt_dir}")
    exit()


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate transcriptions")
    parser.add_argument(
        "predictions",
        type=Path,
        help=".csv file with predicted transcriptions",
    )
    parser.add_argument(
        "gt_transcriptions",
        type=Path,
        help="The directory containing ground truth transcriptions (.txt-files)",
    )
    parser.add_argument(
        "--model_name",
        help="Name of model that produced transcriptions to evaluate",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="The output directory to store evaluation results",
        default=Path("output/evaluation/"),
    )
    parser.add_argument(
        "--line",
        action="store_true",
        help="If flagged, will assume line level predictions (and page level if not flagged)",
    )
    parser.add_argument(
        "--map_transkribus",
        action="store_true",
        help="If flagged, will map transkribus line bboxes to closest ground truth bboxes",
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
    df["transcription"] = df.transcription.apply(str)

    if args.map_transkribus:
        df["image"] = map_transkribus_image_lines_to_gt_image_lines(
            transkribus_df=df, gt_image_dir=args.gt_transcriptions
        )

    df["langcode"] = df.image.apply(lambda x: Path(x).stem).apply(urn_to_langcode)

    ground_truth_paths = df.image.apply(Path).apply(
        partial(find_gt, gt_dir=args.gt_transcriptions)
    )
    df["ground_truth"] = ground_truth_paths.apply(lambda p: p.read_text())

    if args.line:
        output_dir = args.output_dir / "line_level" / args.model_name
    else:
        output_dir = args.output_dir / "page_level" / args.model_name
    output_dir.mkdir(parents=True)

    # Calculate WER and CER for the entire collection
    coll_wer, coll_cer = evaluate_collection_level(df)
    with (output_dir / "all_rows.json").open("w+") as f:
        f.write(json.dumps({"CER": coll_cer, "WER": coll_wer}))

    # Calculate WER and CER for each oage
    df = evaluate_each_row(df)
    df.to_csv(output_dir / "row_level.csv", index=False)

    if not args.line:
        # Calculate WER and CER for each line in each text
        (output_dir / "line_level").mkdir()
        for e in df.itertuples():
            df = evaluate_line_level(
                ground_truth=e.ground_truth, predicted_transcription=e.transcription
            )
            df.to_csv(output_dir / "line_level" / f"{e.image}.csv", index=False)

    logger.info(f"See WER and CER scores in {output_dir}")
