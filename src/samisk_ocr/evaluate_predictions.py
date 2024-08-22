import json
import logging
from argparse import ArgumentParser
from collections import Counter
from functools import partial
from pathlib import Path
from string import punctuation, whitespace

import pandas as pd
from jiwer import cer, wer

from samisk_ocr.map_transkribus_lines_to_gt_lines import (
    map_transkribus_image_lines_to_gt_image_lines,
)
from samisk_ocr.utils import setup_logging

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


def get_language_specific_chars(base_model_language: str) -> list[str]:
    match base_model_language:
        case "nor":
            base_language_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÅÆØåæø"
        case "eng":
            base_language_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        case "est":
            base_language_alphabet = (
                "ABDEFFGHIJKLMNOPRSTUVZZabdeffghijklmnoprstuvzzÄÕÖÜäõöüŠŠššŽŽžž"
            )
        case "fin":
            base_language_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÅÄÖåäö"
        case _:
            logger.warning(
                f"No alphabet found for base model language {base_model_language}, using default alphabet (eng)"
            )
            base_language_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    test_chars = Path("data/testset_characters.txt").read_text()
    not_letters = punctuation + whitespace + "«»–§"
    special_letters = [
        c for c in test_chars if c not in base_language_alphabet + not_letters and not c.isnumeric()
    ]
    return special_letters


def evaluate_collection_level(df: pd.DataFrame, special_chars: list[str] = []) -> dict[str, float]:
    """Calculate WER and CER across rows in df"""

    scores = {}
    scores["WER"] = wer(reference=df.ground_truth.to_list(), hypothesis=df.transcription.to_list())
    scores["CER"] = cer(reference=df.ground_truth.to_list(), hypothesis=df.transcription.to_list())

    if special_chars:
        reference_counters = df.ground_truth.apply(Counter)
        hypothesis_counters = df.transcription.apply(Counter)
        for char in special_chars:
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            scores[char] = {}
            for ref_counter, hyp_counter in zip(reference_counters, hypothesis_counters):
                ref_count = ref_counter[char]
                hyp_count = hyp_counter[char]
                if ref_count == hyp_count:
                    true_positives += ref_count
                    continue
                if ref_count > hyp_count:
                    # not all occurences of char have been correctly transcribed
                    false_negatives += ref_count - hyp_count
                    true_positives += hyp_count
                if hyp_count > ref_count:
                    # char has been transcribed when it wasn't there
                    false_positives += hyp_count - ref_count
                    true_positives += ref_count
            if true_positives + false_positives == 0:
                # The model never predicted the character
                scores[char]["Precision"] = 0
            else:
                scores[char]["Precision"] = true_positives / (true_positives + false_positives)
            scores[char]["Recall"] = true_positives / (true_positives + false_negatives)
            scores[char]["F1"] = (2 * true_positives) / (
                2 * true_positives + false_positives + false_negatives
            )

    return scores


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


def evaluate_line_level(ground_truth: str, predicted_transcription: str) -> pd.DataFrame:
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


def get_parser() -> ArgumentParser:
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
        "--base_model_language",
        default="",
        help="Three-letter langcode for language for the base model (if any)",
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
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    setup_logging(source_script="evaluate_predictions", log_level=args.log_level)

    df = pd.read_csv(args.predictions)
    df["transcription"] = df.transcription.apply(str)

    if args.map_transkribus:
        df["image"] = map_transkribus_image_lines_to_gt_image_lines(
            transkribus_df=df, gt_image_dir=args.gt_transcriptions
        )

    df["langcode"] = df.image.apply(lambda x: Path(x).stem).apply(urn_to_langcode)

    ground_truth_paths = df.image.apply(Path).apply(partial(find_gt, gt_dir=args.gt_transcriptions))
    df["ground_truth"] = ground_truth_paths.apply(lambda p: p.read_text())

    if args.line:
        output_dir = args.output_dir / "line_level" / args.model_name
    else:
        output_dir = args.output_dir / "page_level" / args.model_name
    output_dir.mkdir(parents=True)

    special_chars = []
    if args.base_model_language:
        special_chars = get_language_specific_chars(args.base_model_language)

    # Calculate WER and CER for the entire collection
    coll_scores = evaluate_collection_level(df, special_chars=special_chars)
    with (output_dir / "all_rows.json").open("w+") as f:
        f.write(json.dumps(coll_scores, ensure_ascii=False, indent=4))

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
