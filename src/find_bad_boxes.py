from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
from utils import setup_logging, clean_transcriptions
from Levenshtein import distance
import logging
from shutil import copy2

logger = logging.getLogger(__name__)


def first_char_different(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df.ground_truth.apply(lambda s: s[0]) != df.transcription.apply(lambda s: s[0])
    ]


def last_char_different(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df.ground_truth.apply(lambda s: s[-1])
        != df.transcription.apply(lambda s: s[-1])
    ]


def relative_edit_distance_too_big(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    distances = df.apply(
        lambda row: distance(row.ground_truth, row.transcription), axis=1
    )
    relative_distances = distances / df.ground_truth.apply(len)
    return df[relative_distances > threshold]


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Transcribe with tesseract model")
    parser.add_argument(
        "csv",
        help=".csv file with predicted transcriptions and ground truth",
    )
    parser.add_argument(
        "threshold",
        type=float,
        default=0.2,
        help="Threshold for relative edit distance too big",
    )
    parser.add_argument(
        "--copy_lines",
        action="store_true",
        help="If flagged, will copy problematic lines to output directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to store copied lines",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    return parser


def copy_lines(df_map: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Copy line images and ground truth transcriptions from dataframes to output dir"""
    data_p = Path("../data/")

    for dirname, df in df_map.items():
        subdir = output_dir / dirname
        subdir.mkdir(parents=True)

        for e in df.itertuples:
            img_file = next(data_p.glob(f"*/{e.image}"))
            copy2(src=img_file, dst=subdir / img_file.name)

        df.to_csv(subdir / "line_data.csv", index=False)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    setup_logging(source_script="tesseract_transcribe", log_level=args.log_level)

    if args.copy_lines and not args.output_dir:
        logger.error("If copy_lines is flagged, output_dir must be provided")
        exit()

    df = pd.read_csv(args.csv)
    prev_len = len(df)
    df = df.dropna()
    logger.info(
        f"Number of lines without prediction: {prev_len-len(df)} or {round((prev_len-len(df))/prev_len*100, 2)}%"
    )

    df["transcription"] = clean_transcriptions(df.transcription)

    first_char_diff_df = first_char_different(df)
    last_char_diff_df = last_char_different(df)
    edit_distance_big_df = relative_edit_distance_too_big(df, args.threshold)

    logger.info(
        f"Number of lines with first char different:            {len(first_char_diff_df)} or {round((len(first_char_diff_df)/len(df)*100),2)}%"
    )
    logger.info(
        f"Number of lines with last char different:             {len(last_char_diff_df)} or {round((len(last_char_diff_df)/len(df)*100),2)}%"
    )
    logger.info(
        f"Number of lines with relative edit distance too big:  {len(edit_distance_big_df)} or {round((len(edit_distance_big_df)/len(df)*100),2)}%"
    )

    if args.copy_lines:
        logger.info(f"Copying problematic lines into {args.output_dir}")
        df_map = {
            "first_char_different": first_char_diff_df,
            "last_char_different": last_char_diff_df,
            "edit_distance_big": edit_distance_big_df,
        }

        copy_lines(df_map=df_map, output_dir=args.output_dir)
