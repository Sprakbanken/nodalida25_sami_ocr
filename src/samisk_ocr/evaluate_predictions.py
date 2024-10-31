from collections.abc import Sequence
import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

import pandas as pd
import stringalign
from stringalign.evaluation import TranscriptionEvaluator

from samisk_ocr.metrics import SpecialCharacterF1, compute_cer, compute_wer
from samisk_ocr.utils import langcodes_to_langcode, setup_logging
from samisk_ocr.clean_text_data import clean

logger = logging.getLogger(__name__)


def get_language_specific_chars(lang: Literal["smi", "sme", "smn", "sma", "smj"]) -> list[str]:
    """Get language specific chracters for a Sámi language"""
    char_df = pd.read_csv("data/common/samiske_bokstaver_med_sprak.csv")
    if lang == "smi":
        char_df = char_df.query("not (eng or nor) and (sme or sma or smj or smn)")

    else:
        char_df = char_df.query(f"{lang} and not (eng or nor)")
    return char_df.bokstav.to_list()


def get_character_info(
    ground_truth: Sequence[str], transcription: Sequence[str]
) -> dict[str, tuple[tuple[tuple[str, str], int], ...] | tuple[tuple[str, int], ...]]:
    evaluator = TranscriptionEvaluator.from_strings(
        ground_truth,
        transcription,
    )
    aggregated_confusion_matrix = sum(
        (
            stringalign.statistics.StringConfusionMatrix.from_strings_and_alignment(
                reference=le.reference, predicted=le.predicted, alignment=le.alignment
            )
            for le in evaluator.line_errors
        ),
        start=stringalign.statistics.StringConfusionMatrix.get_empty(),
    )
    return {
        "mistakes": tuple(
            ((op.generalize().substring, op.generalize().replacement), count)
            for op, count in aggregated_confusion_matrix.edit_counts.most_common()
        ),
        "true_positives": tuple(evaluator.confusion_matrix.true_positives.items()),
        "false_positives": tuple(evaluator.confusion_matrix.false_positives.items()),
        "false_negatives": tuple(evaluator.confusion_matrix.false_negatives.items()),
    }


def evaluate(df: pd.DataFrame, output_dir: Path):
    df["ground_truth"] = df.ground_truth.apply(clean)
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

    collection_level_scores |= get_character_info(
        ground_truth=df["ground_truth"], transcription=df["transcription"]
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

        with (output_dir / f"{lang}_rows.json").open("w+", encoding="utf-8") as f:
            f.write(json.dumps(lang_scores, indent=4))

    df.to_csv(output_dir / "row_level.csv", index=False)

    with (output_dir / "all_rows.json").open("w+", encoding="utf-8") as f:
        f.write(json.dumps(collection_level_scores, indent=4))


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
        "--remove_pliktmono",
        action="store_true",
        help="If flagged, will remove rows where 'pliktmonografi' is a substring in column 'file_name' ",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument(
        "--baseline_csv",
        type=Path,
        help="Path to csv with baseline transcriptions, used to filter out rows where the baseline has CER > 0.5",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    setup_logging(source_script="evaluate_predictions", log_level=args.log_level)
    logger.info(args)

    if args.baseline_csv:
        baseline_df = pd.read_csv(args.baseline_csv)
        baseline_df["cer"] = baseline_df.apply(
            lambda row: compute_cer(ground_truth=row.text, transcription=row.transcription),
            axis=1,
        )
        to_skip = set(Path(n).name for n in baseline_df.query("cer > 0.5").file_name)
    else:
        to_skip = set()

    logger.info("Skipping %s rows due to bad baseline: %s", len(to_skip), to_skip)
    for prediction in args.prediction_dir.iterdir():
        df = pd.read_csv(prediction)
        df = df.assign(file_name=df.file_name.map(lambda x: Path(x).name))
        df = df.query("file_name not in @to_skip", local_dict={"to_skip": to_skip})

        if args.remove_pliktmono:
            logger.info("Removing pliktmonografi rows")
            logger.debug("Num rows before %s", len(df))
            df = df[df.file_name.apply(lambda x: "pliktmonografi" not in x)]
            df.index = range(len(df))
            logger.debug("Num rows after %s", len(df))

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
