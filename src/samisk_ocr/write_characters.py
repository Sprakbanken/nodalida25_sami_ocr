from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from samisk_ocr.utils import setup_logging


def get_text(df: pd.DataFrame, text_column: str = "text") -> str:
    all_text = ""
    for text in df[text_column]:
        all_text += f" {text}"
    return all_text


def get_chars(df: pd.DataFrame, text_column: str = "text") -> str:
    text = get_text(df, text_column=text_column)
    return "".join(sorted(list(set(text))))


def write_characters_all_splits(dataset_dir: Path, output_dir: Path):
    for e in dataset_dir.iterdir():
        if e.is_dir():
            split_name = e.name
            metadata_csv = e / "metadata.csv"
            if metadata_csv.exists():
                df = pd.read_csv(metadata_csv)
                chars = get_chars(df)
                with Path(output_dir / f"{split_name}set_characters.txt").open("w+") as f:
                    f.write(chars)


if __name__ == "__main__":
    parser = ArgumentParser(description="Write dataset characters to file")
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to dataset",
        default=Path("data/samisk_ocr_dataset/val"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="The output directory to store characters",
        default=Path("data/"),
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    args = parser.parse_args()
    setup_logging(source_script="tesseract_transcribe", log_level=args.log_level)

    write_characters_all_splits(args.dataset, args.output_dir)
