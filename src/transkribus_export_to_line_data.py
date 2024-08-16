from ordbilder import create_cropped_image
from ordbilder.annotations import get_annotation_information, Annotation
from pathlib import Path
from PIL import Image
import pandas as pd
import logging
from utils import setup_logging

from argparse import ArgumentParser
from shutil import copy2, rmtree


pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def get_annotations_for_image(image_path: Path | str) -> list[Annotation]:
    image_path = Path(image_path)
    xml_path = Path(image_path.parent / "alto" / f"{image_path.stem}.xml")
    return get_annotation_information(xml_path)


def transkribus_export_to_words_pages(
    base_image_dir: Path,
    output_dir: Path,
    img_suffixes: list[str] = [".jpg", ".png", ".tif"],
) -> None:
    """Adapted from Marie Roalds original code:
    https://github.com/Sprakbanken/hugin-munin-ordbilder/blob/main/scripts/create_unadjusted_dataset.py [Accessed: Aug 7, 2024].

    Modifications:
    - Remove auhor information
    - Add rows even if image exists
    """
    rows = []
    images = [e for suf in img_suffixes for e in base_image_dir.glob(f"**/*{suf}")]
    for input_image_path in sorted(images):
        logger.info(f"Creating output for {input_image_path}")
        annotations = get_annotations_for_image(input_image_path)

        # Split the filename into parts
        parent_dir = input_image_path.parent.parent.name
        image_name = input_image_path.stem
        image_dir = input_image_path.parent.name

        # Create the output directory if it does not exist
        output_word_subdir = output_dir / "words" / parent_dir / image_dir
        output_word_subdir.mkdir(exist_ok=True, parents=True)
        output_page_subdir = output_dir / "pages" / parent_dir / image_dir
        output_page_subdir.mkdir(exist_ok=True, parents=True)
        for i, annotation in enumerate(annotations):
            # Extract the bounding box from the annotation
            bbox = annotation["bbox"]
            x1, y1, x2, y2 = bbox
            # Create the output image path
            output_image_path = (
                output_word_subdir
                / f"{image_name}_{i:03d}_{x1:04d}_{y1:04d}_{x2:04d}_{y2:04d}.jpg"
            )
            output_page_path = output_page_subdir / input_image_path.name

            if output_image_path.exists():
                logger.debug(f"{output_image_path} exists, skipping")
                row = {
                    "page_image": str(output_page_path.relative_to(output_dir)),
                    "word_image": str(output_image_path.relative_to(output_dir)),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "word": annotation["word"],
                }
                rows.append(row)
                continue

            # Create the page image
            output_page_path.parent.mkdir(exist_ok=True, parents=True)
            copy2(input_image_path, output_page_path)
            try:
                # Create the cropped image
                create_cropped_image(
                    input_image_path,
                    output_image_path,
                    bbox,
                    quality=100,
                )
                logger.debug(f"Created {output_image_path}")
            except Exception as ex:
                logger.warn(ex)
                logger.info(input_image_path, i)
                logger.info("----")
                continue

            row = {
                "page_image": str(output_page_path.relative_to(output_dir)),
                "word_image": str(output_image_path.relative_to(output_dir)),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "word": annotation["word"],
            }
            rows.append(row)

        dataset_df = pd.DataFrame(rows)
        dataset_df.to_csv(output_dir / "metadata.csv", index=False)


def words_pages_to_lines(
    source_data_dir: Path,
    destination_data_dir: Path,
    txt_suffix: str,
    rotate: bool,
    remove: bool,
):
    """Create .tif and (.gt).txt line level transcriptions from ordbilder output"""
    metadata_df = pd.read_csv(source_data_dir / "metadata.csv")

    copied = 0
    rotated = 0
    removed = 0

    for e in metadata_df.itertuples():
        line_image_path = source_data_dir / e.word_image
        assert line_image_path.exists()
        img = Image.open(line_image_path)
        width, height = img.size
        if rotate and width * 2 < height:
            logger.debug(f"Rotating image {e.word_image}")
            rotated += 1
            img = img.rotate(-90, expand=True)
        elif remove and width < height:
            logger.debug(
                f"Skipping image {e.word_image} (width to height ratio too small)"
            )
            removed += 1
            continue
        file_stem = line_image_path.stem
        new_image_path = destination_data_dir / f"{file_stem}.tif"

        img.save(new_image_path)
        copied += 1

        # convert numbers to strings and replace non-breaking space character with normal space
        transcription = str(e.word).replace("\xa0", " ")
        text_path = destination_data_dir / f"{file_stem}{txt_suffix}"
        with text_path.open("w+") as f:
            f.write(transcription)
    logger.info(
        f"Copied {copied} line images and transcriptions to {destination_data_dir}"
    )
    logger.info(f"{rotated} images were rotated due to width/height ratio")
    logger.info(f"{removed} images were skipped due to width/height ratio")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Create .tif line images and .txt line transcriptions from a transkribus export"
    )
    parser.add_argument(
        "base_image_dir",
        type=Path,
        help="The directory containing .jpg images and alto/ directory with xml files ",
    )
    parser.add_argument("output_dir", type=Path, help="The output directory")
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument(
        "--keep_temp_dir",
        action="store_true",
        help="If flagged, keep <output_dir>/temp with pages, words and metadata.csv",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="If flagged, rotate images where width is less than half of height",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="If flagged, remove files where width is less than height",
    )
    parser.add_argument(
        "--gt",
        type=int,
        default=1,
        help="If 1, will save .txt files as .gt.txt (only .txt if 0)",
    )
    args = parser.parse_args()
    setup_logging(source_script="transkribus_data_to_lines", log_level=args.log_level)

    temp_dir = args.output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created output directory in {args.output_dir}")

    transkribus_export_to_words_pages(
        base_image_dir=args.base_image_dir, output_dir=temp_dir
    )
    words_pages_to_lines(
        source_data_dir=temp_dir,
        destination_data_dir=args.output_dir,
        txt_suffix=".gt.txt" if args.gt else ".txt",
        rotate=args.rotate,
        remove=args.remove,
    )

    if not args.keep_temp_dir:
        rmtree(temp_dir)
