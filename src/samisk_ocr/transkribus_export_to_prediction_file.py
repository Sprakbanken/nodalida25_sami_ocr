import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from samisk_ocr.utils import setup_logging

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def read_text_files(image_dir: Path, df: pd.DataFrame, line: bool) -> pd.DataFrame:
    texts = []
    text_dir = image_dir / "txt"
    if line:
        text_dir = image_dir

    for e in df.itertuples():
        filename_stem = Path(e.image).stem
        text_file = text_dir / f"{filename_stem}.txt"
        if not text_file.exists():
            logger.error(f"Could not find textfile for image {e.image}")
            exit()
        texts.append(text_file.read_text())
    df["transcription"] = texts
    return df


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Create transcription file from transkribus export")
    parser.add_argument(
        "model_name",
        help="Name of transkribus model",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="The directory containing image files and txt/ directory with transcription files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="The output directory to store predicted transcriptions",
        default="output/predictions",
    )
    parser.add_argument(
        "--line",
        action="store_true",
        help="If flagged, look for line level transcriptions in same directory as images",
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
    setup_logging(source_script="transkribus_export_to_prediction_file", log_level=args.log_level)
    if args.line:
        output_dir = args.output_dir / "line_level"
    else:
        output_dir = args.output_dir / "page_level"

    output_dir.mkdir(exist_ok=True, parents=True)

    images = [img.name for img in args.input_dir.glob("*.jpg")] + [
        img.name for img in args.input_dir.glob("*.tif")
    ]

    df = pd.DataFrame({"image": images})
    df["model_name"] = [args.model_name] * len(df)
    df = read_text_files(df=df, image_dir=args.input_dir, line=args.line)

    output_csv = output_dir / f"{args.model_name}_predictions.csv"

    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote predicted transcriptions to {output_csv}")
