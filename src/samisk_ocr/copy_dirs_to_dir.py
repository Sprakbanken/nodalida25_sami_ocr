import logging
from argparse import ArgumentParser
from pathlib import Path
from shutil import copytree

from samisk_ocr.utils import setup_logging

logger = logging.getLogger(__name__)


def has_overlapping_subdirs(input_dir: Path, output_dir: Path) -> bool:
    existing_files = [e.name for e in output_dir.iterdir()]
    for sub_dir in input_dir.iterdir():
        if sub_dir.name in existing_files:
            return True
    return False


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to directory (e.g. transkribus export job directory with subdirectories with alto/, page/ and metadata.xml)",
    )
    parser.add_argument("output_dir", type=Path, help="Path to directory to copy to")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If flagged, will overwrite any files that already exist",
    )
    args = parser.parse_args()
    setup_logging(source_script="copy_dirs_to_dir", log_level="DEBUG")

    if not args.overwrite and has_overlapping_subdirs(
        input_dir=args.input_dir, output_dir=args.output_dir
    ):
        logger.error(f"Some dirs already exists in {args.output_dir}, but overwrite is not flagged")
        exit(1)

    for sub_dir in args.input_dir.glob("*/"):
        copytree(sub_dir, dst=args.output_dir / sub_dir.name, dirs_exist_ok=True)
