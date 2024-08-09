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


def transkribus_export_to_words_pages(base_image_dir: Path, output_dir: Path) -> None:
    """Adapted from Marie Roalds original code:
    https://github.com/Sprakbanken/hugin-munin-ordbilder/blob/main/scripts/create_unadjusted_dataset.py [Accessed: Aug 7, 2024].

    Modifications:
    - Remove author information
    """
    rows = []
    for input_image_path in sorted(base_image_dir.glob("**/*.jpg")):
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
            if output_image_path.exists():
                logger.info(f"{output_image_path} exists, skipping")
                continue
            # Create output page path
            output_page_path = output_page_subdir / input_image_path.name

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


def words_pages_to_lines(source_data_dir: Path, destination_data_dir: Path):
    """Create .tif and .gt.txt line level transcriptions from ordbilder output"""
    metadata_df = pd.read_csv(source_data_dir / "metadata.csv")

    for e in metadata_df.itertuples():
        line_image_path = source_data_dir / e.word_image
        assert line_image_path.exists()

        file_stem = line_image_path.stem
        new_image_path = destination_data_dir / f"{file_stem}.tif"

        img = Image.open(line_image_path)
        img.save(new_image_path)

        transcription = e.word
        transcription = transcription.replace("\xa0", " ")
        text_path = destination_data_dir / f"{file_stem}.gt.txt"
        with text_path.open("w+") as f:
            f.write(transcription)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Create .tif line images and .gt.txt line transcriptions from a transkribus export"
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
    args = parser.parse_args()

    setup_logging(source_script="transkribus_data_to_lines", log_level=args.log_level)

    temp_dir = args.output_dir / "temp"
    temp_dir.mkdir(parents=True)

    logger.info(f"Created output directory in {args.output_dir}")

    transkribus_export_to_words_pages(
        base_image_dir=args.base_image_dir, output_dir=temp_dir
    )

    words_pages_to_lines(source_data_dir=temp_dir, destination_data_dir=args.output_dir)

    if not args.keep_temp_dir:
        rmtree(temp_dir)
