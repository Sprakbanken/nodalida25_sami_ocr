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


def transcribe(model_name: str, image: Path, config: str) -> str:
    return pytesseract.image_to_string(
        Image.open(image), lang=model_name, config=config
    )


def transcribe_dataset(
    model_name: str,
    dataset: Path,
    split: str,
    line_level: bool,
) -> pd.DataFrame:
    """Run tesseract model on all images in dataset split"""

    df = pd.read_csv(dataset / split / "_metadata.csv")
    tesseract_config = "--psm 7" if line_level else ""

    transcriptions = []

    for tup in tqdm(df.itertuples(), total=len(df)):
        img_file_name = dataset / split / str(tup.file_name)
        transcription = transcribe(
            model_name=model_name,
            image=img_file_name,
            config=tesseract_config,
        )
        if line_level and not transcription:
            logger.debug("No transcription for %s", tup.file_name)
            logger.debug(
                "Trying to transcribe with --psm 8 (treat image as single word)"
            )
            transcription = transcribe(
                model_name, image=img_file_name, config="--psm 8"
            )
        if not transcription:
            logger.debug("No transcription for %s", tup.file_name)
        transcriptions.append(transcription)

    df["transcription"] = transcriptions
    return df


if __name__ == "__main__":
    parser = ArgumentParser(description="Transcribe with tesseract model")
    parser.add_argument(
        "model_name",
        help="Name of tesseract model (must be in list of available models with 'tesseract --list-langs')",
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
        default=Path("output/predictions/"),
    )
    parser.add_argument(
        "--page",
        action="store_true",
        help="If flagged, will treat images as page level",
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

    df = transcribe_dataset(
        model_name=args.model_name,
        dataset=args.dataset,
        split=args.split,
        line_level=not args.page,
    )

    if args.page:
        output_dir = args.output_dir / "page_level"
    else:
        output_dir = args.output_dir / "line_level"

    output_dir.mkdir(exist_ok=True, parents=True)
    output_csv = output_dir / f"{args.model_name}_{args.split}.csv"
    df.to_csv(output_csv, index=False)

    logger.info(f"Wrote predicted transcriptions to {output_csv}")
