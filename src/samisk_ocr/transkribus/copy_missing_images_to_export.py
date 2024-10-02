import logging
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy2

from samisk_ocr.utils import setup_logging

logger = logging.getLogger(__name__)


def find_img_file(img_dir: Path, img_stem: str, suffixes=[".jpg", ".jpeg", ".tif", ".png"]) -> Path:
    for suffix in suffixes:
        img = img_dir / f"{img_stem}{suffix}"
        if img.exists():
            return img
    logger.error("No image found with stem %s found in %s", img_stem, img_dir)
    exit(1)


def get_image_files_for_alto_xml_files(alto_dir: Path, export_with_images: Path) -> list[Path]:
    xml_files = alto_dir.glob("*.xml")
    images = []
    for xml_file in xml_files:
        full_export_xml_twin = next(export_with_images.glob(f"**/{xml_file.name}"))
        image_file = find_img_file(full_export_xml_twin.parent.parent, img_stem=xml_file.stem)
        images.append(image_file)
    return images


if __name__ == "__main__":
    parser = ArgumentParser("Copy missing image files to a transkribus export with only xml")
    parser.add_argument("only_xml_export", type=Path, help="Transkribus export with missing images")
    parser.add_argument(
        "full_export", type=Path, help="Transkribus export containing images", default=Path("data/")
    )
    args = parser.parse_args()

    setup_logging(source_script="copy_missing_images_to_export", log_level="DEBUG")

    alto_dirs = args.only_xml_export.glob("**/alto/")
    for alto_dir in alto_dirs:
        images = get_image_files_for_alto_xml_files(
            alto_dir=alto_dir, export_with_images=args.full_export
        )
        for img in images:
            copy2(src=img, dst=alto_dir.parent)
