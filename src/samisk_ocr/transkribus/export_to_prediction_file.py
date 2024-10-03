import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from ordbilder.annotations import Annotation, get_annotation_information

from samisk_ocr.transkribus.map_transkribus_lines_to_gt_lines import (
    map_transkribus_image_lines_to_gt_image_lines,
)
from samisk_ocr.utils import setup_logging

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def get_images_in_dir(image_dir: Path, image_suffixes=(".tif", ".png", ".jpg")):
    return [e for suf in image_suffixes for e in image_dir.glob(f"*{suf}")]


def read_text_files(text_dir: Path, df: pd.DataFrame) -> list[str]:
    text_files = [text_dir / f"{Path(image).stem}.txt" for image in df.image]
    if not all([text_file.exists() for text_file in text_files]):
        logger.error("Could not find a textfile in %s for every image in dataframe", text_dir)
        exit(1)

    return [e.read_text() for e in text_files]


def image_stem_annotation_to_filename(
    image_stem: str, annotation: Annotation, line_num: int
) -> str:
    x1, y1, x2, y2 = annotation["bbox"]
    return f"{image_stem}_{line_num:03d}_{x1:04d}_{y1:04d}_{x2:04d}_{y2:04d}.jpg"


def get_line_transcriptions(input_dir: Path) -> pd.DataFrame:
    """Get line transcriptions from transkribus xml-file, and create 'filenames' based on xml-file-bbox"""
    alto_xml_files = input_dir.glob("**/alto/*.xml")
    df_data = {"image": [], "transcription": []}
    for alto_xml_file in alto_xml_files:
        annotations = get_annotation_information(alto_xml_file)
        df_data["transcription"] += [annotation["word"] for annotation in annotations]
        df_data["image"] += [
            image_stem_annotation_to_filename(alto_xml_file.stem, annotation=annotation, line_num=i)
            for i, annotation in enumerate(annotations)
        ]
    return pd.DataFrame(df_data)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Create transcription file from transkribus export")
    parser.add_argument(
        "model_name",
        help="Name of transkribus model",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="The directory containing transkribus transcriptions (alto_xml files if line-level or txt/ directory if page-level)",
    )
    parser.add_argument(
        "--dataset",
        help="Path to local dataset",
        type=Path,
        default=Path("data/samisk_ocr_dataset/"),
    )
    parser.add_argument("--split", default="val", help="Dataset split to transcribe")
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="The output directory to store predicted transcriptions",
        default="output/predictions",
    )
    parser.add_argument(
        "--page",
        action="store_true",
        help="If flagged, will treat images as page level (default is line level)",
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
    logger.info(args)

    if args.page:
        images = get_images_in_dir(args.input_dir)
        df = pd.DataFrame({"image": images})
        df["transcription"] = read_text_files(df=df, text_dir=args.input_dir / "txt")

        output_dir = args.output_dir / "page_level"
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        df = get_line_transcriptions(args.input_dir)
        output_dir = args.output_dir / "line_level"
        output_dir.mkdir(exist_ok=True, parents=True)

        df["gt_image"] = map_transkribus_image_lines_to_gt_image_lines(
            df, gt_image_dir=args.dataset / args.split
        )
        dataset_df = pd.read_csv(args.dataset / args.split / "_metadata.csv")
        df = dataset_df.merge(
            df[["transcription", "gt_image"]],
            left_on="file_name",
            right_on="gt_image",
            validate="1:1",
        )
        assert len(df) == len(dataset_df)

    output_csv = output_dir / f"{args.model_name}_{args.split}.csv"
    df.to_csv(output_csv, index=False)

    logger.info(f"Wrote predicted transcriptions to {output_csv}")
