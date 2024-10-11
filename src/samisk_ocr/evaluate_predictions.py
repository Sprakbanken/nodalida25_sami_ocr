import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

import pandas as pd

from samisk_ocr.metrics import SpecialCharacterF1, compute_cer, compute_wer
from samisk_ocr.utils import langcodes_to_langcode, setup_logging

logger = logging.getLogger(__name__)


def get_language_specific_chars(lang: Literal["smi", "sme", "smn", "sma", "smj"]) -> list[str]:
    """Get language specific chracters for a Sámi language"""
    char_df = pd.read_csv("data/common/samiske_bokstaver_med_sprak.csv")
    if lang == "smi":
        char_df = char_df.query("not (eng or nor) and (sme or sma or smj or smn)")

    else:
        char_df = char_df.query(f"{lang} and not (eng or nor)")
    return char_df.bokstav.to_list()


def evaluate(df: pd.DataFrame, output_dir: Path):
    df["CER"] = df.apply(
        lambda row: compute_cer(transcription=row.transcription, ground_truth=row.ground_truth),
        axis=1,
    )
    df["WER"] = df.apply(
        lambda row: compute_wer(transcription=row.transcription, ground_truth=row.ground_truth),
        axis=1,
    )

    collection_level_scores = {}
    collection_level_scores["WER_mean"] = df.WER.mean()
    collection_level_scores["CER_mean"] = df.CER.mean()
    collection_level_scores["WER_concat"] = compute_wer(
        transcription=" ".join(df.transcription), ground_truth=" ".join(df.ground_truth)
    )
    collection_level_scores["CER_concat"] = compute_cer(
        transcription="".join(df.transcription), ground_truth="".join(df.ground_truth)
    )

    special_chars = get_language_specific_chars(lang="smi")
    general_scorer = SpecialCharacterF1("".join(special_chars))

    df["special_char_F1"] = df.apply(
        lambda row: general_scorer(transcription=row.transcription, ground_truth=row.ground_truth),
        axis=1,
    )
    collection_level_scores["special_char_F1_mean"] = df.special_char_F1.mean()
    collection_level_scores["special_char_F1_concat"] = general_scorer(
        ground_truth="".join(df.ground_truth), transcription="".join(df.transcription)
    )

    for char in special_chars:
        char_scorer = SpecialCharacterF1(char)
        df[f"{char}_F1"] = df.apply(
            lambda row: char_scorer(transcription=row.transcription, ground_truth=row.ground_truth),
            axis=1,
        )
        collection_level_scores[char] = {"F1_mean": df[f"{char}_F1"].mean()}
        collection_level_scores[char]["F1_concat"] = char_scorer(
            transcription="".join(df.transcription), ground_truth="".join(df.ground_truth)
        )

    for lang, lang_df in df.groupby("langcode"):
        lang_scores = {}
        lang_scores["WER_mean"] = lang_df.WER.mean()
        lang_scores["CER_mean"] = lang_df.CER.mean()
        lang_scores["WER_concat"] = compute_wer(
            transcription=" ".join(lang_df.transcription),
            ground_truth=" ".join(lang_df.ground_truth),
        )
        lang_scores["CER_concat"] = compute_cer(
            transcription="".join(lang_df.transcription), ground_truth="".join(lang_df.ground_truth)
        )

        lang_scores["special_char_F1_mean"] = lang_df.special_char_F1.mean()

        special_chars = get_language_specific_chars(lang=lang)
        lang_scorer = SpecialCharacterF1("".join(special_chars))
        lang_scores["special_char_F1_concat"] = lang_scorer(
            ground_truth="".join(lang_df.ground_truth), transcription="".join(lang_df.transcription)
        )

        for char in special_chars:
            lang_scores[char] = {"F1_mean": lang_df[f"{char}_F1"].mean()}

            char_scorer = SpecialCharacterF1(char)
            lang_scores[char]["F1_concat"] = char_scorer(
                ground_truth="".join(lang_df.ground_truth),
                transcription="".join(lang_df.transcription),
            )

        with (output_dir / f"{lang}_rows.json").open("w+") as f:
            f.write(json.dumps(lang_scores, ensure_ascii=False, indent=4))

    df.to_csv(output_dir / "row_level.csv", index=False)

    with (output_dir / "all_rows.json").open("w+") as f:
        f.write(json.dumps(collection_level_scores, ensure_ascii=False, indent=4))


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate transcriptions")
    parser.add_argument(
        "prediction_dir",
        type=Path,
        help="Directory containing .csv-files with predicted transcriptions",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="The output directory to store evaluation results",
        default=Path("output/evaluation/"),
    )
    parser.add_argument(
        "--page",
        action="store_true",
        help="If flagged, will assume page level predictions (default is line level)",
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
    logger.info(args)

    for prediction in args.prediction_dir.iterdir():
        df = pd.read_csv(prediction)
        df["transcription"] = df.transcription.apply(str)
        df = df.rename(columns={"text": "ground_truth"})

        df = langcodes_to_langcode(df)

        model_name = prediction.name.rpartition("_")[0]

        if args.page:
            output_dir = args.output_dir / "page_level" / model_name
        else:
            output_dir = args.output_dir / "line_level" / model_name

        output_dir.mkdir(parents=True, exist_ok=True)

        evaluate(df, output_dir=output_dir)

        logger.info(f"See evaluation results for {prediction} in {output_dir}")
