import logging
from argparse import ArgumentParser
from pathlib import Path

from samisk_ocr.utils import setup_logging

logger = logging.getLogger(__name__)


def clean(text: str) -> str:
    bad_chars_and_replacements = [
        # replace non-breaking space with normal space
        ("\xa0", " "),
        # replace variants of quotation marks
        ("«", '"'),
        ("»", '"'),
        ("”", '"'),
        # replace variants of apostrophe
        ("ʼ", "'"),
        ("’", "'"),
        ("ʹ", "'"),
        # replace Ds
        ("Ð", "Đ"),
        ("Ɖ", "Đ"),
        # replace em dash with en dash
        ("—", "–"),
    ]
    for bad_char, replacement in bad_chars_and_replacements:
        text = text.replace(bad_char, replacement)
    return text


def clean_directory_in_place(directory: Path):
    i = 0
    for text_file in directory.glob("**/*.txt"):
        text_pre = text_file.read_text()
        text = clean(text_pre)
        if text_pre != text:
            i += 1
            with text_file.open("w") as f:
                f.write(text)
    logger.debug(f"Cleaned {i} textfiles")


def clean_directory_to_output_dir_only_copy_cleaned(directory: Path, output_directory: Path):
    i = 0
    for text_file in directory.glob("**/*.txt"):
        text_pre = text_file.read_text()
        text = clean(text_pre)
        if text != text_pre:
            i += 1

            out_parent = output_directory / text_file.parent.relative_to(directory)
            out_parent.mkdir(exist_ok=True, parents=True)
            out_file = out_parent / text_file.name

            with out_file.open("w+") as f:
                f.write(text)
    logger.debug(f"Cleaned {i} textfiles")


def clean_directory_to_output_dir(directory: Path, output_directory: Path):
    i = 0
    for text_file in directory.glob("**/*.txt"):
        text_pre = text_file.read_text()
        text = clean(text_pre)

        i += text != text_pre

        out_parent = output_directory / text_file.parent.relative_to(directory)
        out_parent.mkdir(exist_ok=True, parents=True)
        out_file = out_parent / text_file.name

        with out_file.open("w+") as f:
            f.write(text)
    logger.debug(f"Cleaned {i} textfiles")


if __name__ == "__main__":
    parser = ArgumentParser(description="Replace unwanted characters in textfiles in directory")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="The directory containing textfiles",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="If set, will copy clean files to output dir (else cleans input directory in-place)",
    )
    parser.add_argument(
        "--only_copy_cleaned",
        action="store_true",
        help="If flagged, will only copy the files that needed cleaning to output dir (else copies all files)",
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

    if args.output_dir:
        clean_directory_to_output_dir(
            directory=args.input_dir, output_directory=args.output_dir, only_copy_cleaned=args.only_copy_cleaned
        )
        
        if args.only_copy_cleaned:
            msg = "Copied clean versions of the textfiles that needed cleaning in %s to %s"
        else:
            msg = "Copied clean versions of all textfiles in %s to %s"
        logger.info(msg, args.input_dir, args.output_dir)

    else:
        clean_directory_in_place(args.input_dir)
        logger.info(f"Cleaned all textfiles in {args.input_dir}")
