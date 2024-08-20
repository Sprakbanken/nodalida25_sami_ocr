from pathlib import Path
from utils import setup_logging
import logging
from argparse import ArgumentParser

logger = logging.getLogger(__name__)


def clean(text: str) -> str:
    # replace non-breaking space with normal space
    text = text.replace("\xa0", " ")

    # replace em dash with en dash
    return text.replace("—", "—")


def clean_directory(directory: Path):
    for i, text_file in enumerate(directory.glob("*.txt")):
        text_pre = text_file.read_text()
        text = clean(text_pre)
        with text_file.open("w") as f:
            f.write(text)
    logger.debug(f"Cleaned {i} textfiles")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Replace unwanted characters in textfiles in directory"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="The directory containing textfiles",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    args = parser.parse_args()
    setup_logging(source_script="clean_text_data", log_level=args.log_level)

    clean_directory(args.input_dir)

    logger.info(f"Cleaned all textfiles in {args.input_dir}")
