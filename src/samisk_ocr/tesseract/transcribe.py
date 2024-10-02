import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import pytesseract
from PIL import Image
from tqdm import tqdm

from samisk_ocr.utils import setup_logging

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def transcribe(
    model_name: str,
    image_dir: Path,
    line_level: bool,
    img_suffixes: list[str] = [".jpg", ".png", ".tif"],
) -> pd.DataFrame:
    """Run tesseract model on all files in image_dir"""
    transcriptions = {"model_name": [], "image": [], "transcription": []}
    images = [e for suf in img_suffixes for e in image_dir.glob(f"*{suf}")]

    tesseract_config = ""
    if line_level:
        tesseract_config = "--psm 7"

    for img in tqdm(images):
        transcription = pytesseract.image_to_string(
            Image.open(img), lang=model_name, config=tesseract_config
        )
        if line_level and not transcription:
            logger.debug(f"No transcription for {img}")
            logger.debug("Trying to transcribe with --psm 8 (treat image as single word)")
            transcription = pytesseract.image_to_string(
                Image.open(img), lang=model_name, config="--psm 8"
            )

        if not transcription:
            logger.debug(f"No transcription for {img}")
            logger.debug(transcription)

        transcriptions["image"].append(img.name)
        transcriptions["model_name"].append(model_name)
        transcriptions["transcription"].append(transcription)

    df = pd.DataFrame(transcriptions)
    return df


if __name__ == "__main__":
    parser = ArgumentParser(description="Transcribe with tesseract model")
    parser.add_argument(
        "model_name",
        help="Name of tesseract model (must be in list of available models with 'tesseract --list-langs')",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        help="The directory containing images to be transcribed",
        default=Path("data/samisk_ocr_dataset/val"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="The output directory to store predicted transcriptions",
        default=Path("output/predictions/"),
    )
    parser.add_argument(
        "--line",
        action="store_true",
        help="If flagged, will treat images as line level",
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

    df = transcribe(model_name=args.model_name, image_dir=args.image_dir, line_level=args.line)

    if args.line:
        output_dir = args.output_dir / "line_level"
    else:
        output_dir = args.output_dir / "page_level"

    output_dir.mkdir(exist_ok=True, parents=True)

    output_csv = output_dir / f"{args.image_dir.name}_{args.model_name}_predictions.csv"

    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote predicted transcriptions to {output_csv}")
