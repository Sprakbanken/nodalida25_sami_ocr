import logging
from pathlib import Path

import pandas as pd
from ordbilder import create_cropped_image
from ordbilder.annotations import Annotation, get_annotation_information

from samisk_ocr.clean_text_data import clean

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def get_annotations_for_image(image_path: Path | str) -> list[Annotation]:
    image_path = Path(image_path)
    xml_path = Path(image_path.parent / "alto" / f"{image_path.stem}.xml")
    return get_annotation_information(xml_path)


def transkribus_export_to_lines(
    base_image_dir: Path,
    output_dir: Path,
    img_suffixes: list[str] = [".jpg", ".png", ".tif"],
) -> None:
    """Adapted from Marie Roalds original code:
    https://github.com/Sprakbanken/hugin-munin-ordbilder/blob/main/scripts/create_unadjusted_dataset.py [Accessed: Aug 7, 2024].

    Modifications:
    - Remove auhor information
    - Add rows even if image exists
    - Don't copy pages
    - Rename words directory -> lines
    - Rename metadata.csv columns word -> text, word_image -> file_name
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
        output_lines_subdir = output_dir / "lines" / parent_dir / image_dir
        output_lines_subdir.mkdir(exist_ok=True, parents=True)
        for i, annotation in enumerate(annotations):
            # Extract the bounding box from the annotation
            bbox = annotation["bbox"]
            x1, y1, x2, y2 = bbox
            # Create the output image path
            output_image_path = (
                output_lines_subdir
                / f"{image_name}_{i:03d}_{x1:04d}_{y1:04d}_{x2:04d}_{y2:04d}.jpg"
            )

            if output_image_path.exists():
                logger.debug(f"{output_image_path} exists, skipping")
                row = {
                    "file_name": str(output_image_path.relative_to(output_dir)),
                    "text": clean(annotation["word"]),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
                rows.append(row)
                continue

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
                logger.info(f"{input_image_path}, {i}")
                logger.info("----")
                continue

            row = {
                "file_name": str(output_image_path.relative_to(output_dir)),
                "text": clean(annotation["word"]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
            rows.append(row)

    dataset_df = pd.DataFrame(rows)
    dataset_df.to_csv(output_dir / "metadata.csv", index=False)
