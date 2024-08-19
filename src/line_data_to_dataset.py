from pathlib import Path
from PIL import Image
import pandas as pd
import logging
from utils import setup_logging, image_stem_to_urn_page_line_bbox
from datasets import load_dataset

from argparse import ArgumentParser

logger = logging.getLogger(__name__)


def get_size(img: Path) -> tuple[int, int]:
    img = Image.open(img)
    return img.size


def create_dataset_df(
    image_dir: Path,
    image_suffixes: list[str] = [".tif", ".png", ".jpg"],
    text_suffix=".gt.txt",
) -> pd.DataFrame:
    image_paths = pd.Series(
        [e for suf in image_suffixes for e in image_dir.glob(f"**/*{suf}")]
    )
    text_paths = image_paths.apply(lambda x: x.parent / (x.stem + text_suffix))
    if not all(text_paths.apply(lambda x: x.exists())):
        logger.error(
            f"Ground truth files don't exist for all image files (must have same filename except for suffix {text_suffix})"
        )

    df = pd.DataFrame({"file_name": [e.relative_to(image_dir) for e in image_paths]})
    df["text"] = text_paths.apply(lambda x: x.read_text())
    df["text_len"] = text_paths.apply(str).apply(len)

    image_stems = image_paths.apply(lambda x: x.stem)
    urns, pages, lines, bboxes = zip(
        *image_stems.apply(image_stem_to_urn_page_line_bbox)
    )
    df["urn"] = urns
    df["page"] = pages
    df["line"] = lines
    df["bbox"] = bboxes

    df["width"], df["height"] = zip(*image_paths.apply(get_size))
    df["page_30"] = image_paths.apply(lambda x: x.parent.name == "side_30")
    return df.sort_values(["urn", "page", "line"])


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Create a 🤗️ datasets image dataset from line level data"
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="The directory containing to create dataset from",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    args = parser.parse_args()
    setup_logging(source_script="line_data_to_dataset", log_level=args.log_level)

    df = create_dataset_df(image_dir=args.image_dir)
    df.to_csv(args.image_dir / "metadata.csv", index=False)

    dataset = load_dataset("imagefolder", data_dir=args.image_dir)
    logger.info(f"Successfully created 🤗️ image dataset at {args.image_dir}")
