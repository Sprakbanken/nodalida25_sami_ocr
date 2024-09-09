import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from string import punctuation, whitespace

import pandas as pd

from samisk_ocr.map_transkribus_lines_to_gt_lines import (
    map_transkribus_image_lines_to_gt_image_lines,
)
from samisk_ocr.metrics import SpecialCharacterF1, compute_cer, compute_wer
from samisk_ocr.utils import setup_logging
from samisk_ocr.write_characters import get_chars

logger = logging.getLogger(__name__)


def get_language_specific_chars(base_model_language: str, gt_chars: str) -> list[str]:
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

    not_letters = punctuation + whitespace + "«»–§"
    special_letters = [
        c for c in gt_chars if c not in base_language_alphabet + not_letters and not c.isnumeric()
    ]
    return special_letters


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate transcriptions")
    parser.add_argument(
        "predictions",
        type=Path,
        help=".csv file with predicted transcriptions",
    )
    parser.add_argument(
        "--dataset", help="Path to dataset", default=Path("data/samisk_ocr_dataset")
    )
    parser.add_argument("--split", help="Dataset split to evaluate", default="val")
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

    gt_df = pd.read_csv(args.dataset / args.split / "metadata.csv")
    gt_df["image"] = gt_df.file_name.apply(lambda x: Path(x).name)
    df = df.merge(gt_df, on="image")
    df = df.rename(columns={"text": "ground_truth"})

    if args.line:
        output_dir = args.output_dir / "line_level" / args.model_name
    else:
        output_dir = args.output_dir / "page_level" / args.model_name
    output_dir.mkdir(parents=True)

    df["WER"] = df.apply(
        lambda row: compute_cer(transcription=row.transcription, ground_truth=row.ground_truth),
        axis=1,
    )
    df["CER"] = df.apply(
        lambda row: compute_wer(transcription=row.transcription, ground_truth=row.ground_truth),
        axis=1,
    )

    collection_level_scores = {}
    collection_level_scores["WER"] = df.WER.mean()
    collection_level_scores["CER"] = df.CER.mean()
    collection_level_scores["WER_concat"] = compute_wer(
        transcription=" ".join(df.transcription), ground_truth=" ".join(df.ground_truth)
    )
    collection_level_scores["CER_concat"] = compute_cer(
        transcription=" ".join(df.transcription), ground_truth=" ".join(df.ground_truth)
    )

    if args.base_model_language:
        gt_chars = get_chars(df, text_column="ground_truth")

        special_chars = get_language_specific_chars(
            base_model_language=args.base_model_language, gt_chars=gt_chars
        )
        general_scorer = SpecialCharacterF1("".join(special_chars))
        df["special_char_F1"] = df.apply(
            lambda row: general_scorer(
                transcription=row.transcription, ground_truth=row.ground_truth
            ),
            axis=1,
        )
        collection_level_scores["special_char_F1"] = df.special_char_F1.mean()
        for char in special_chars:
            char_scorer = SpecialCharacterF1(char)
            collection_level_scores[char] = {
                "F1": df.apply(
                    lambda row: char_scorer(
                        transcription=row.transcription, ground_truth=row.ground_truth
                    ),
                    axis=1,
                ).mean()
            }

    df.to_csv(output_dir / "row_level.csv", index=False)

    with (output_dir / "all_rows.json").open("w+") as f:
        f.write(json.dumps(collection_level_scores, ensure_ascii=False, indent=4))

    logger.info(f"See evaluation results in {output_dir}")
