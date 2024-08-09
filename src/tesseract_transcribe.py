import pandas as pd
from pathlib import Path
from PIL import Image
import pytesseract
from tqdm import tqdm
import logging
from utils import setup_logging
from argparse import ArgumentParser

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def transcribe(model_name: str, image_dir: Path) -> pd.DataFrame:
    """Run tesseract model on all files in image_dir"""
    transcriptions = {"model_name": [], "image": [], "transcription": []}
    images = list(image_dir.glob("*.jpg"))
    for img in tqdm(images):
        transcription = pytesseract.image_to_string(Image.open(img), lang=model_name)
        if not transcription:
            logger.debug(f"No transcription for {img}")
            logger.debug(transcription)
        transcriptions["image"].append(img.name)
        transcriptions["model_name"].append(model_name)
        transcriptions["transcription"].append(transcription)

    df = pd.DataFrame(transcriptions)
    return df


def find_gt(image_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    texts = []
    for e in df.itertuples():
        filename_stem = Path(e.image).stem
        gt_text_file = image_dir / "txt" / f"{filename_stem}.txt"
        if not gt_text_file.exists():
            logger.warning(f"Could not find ground truth file for image {e.image}")
            texts.append("")
            continue
        texts.append(gt_text_file.read_text())
    df["ground_truth"] = texts
    return df


if __name__ == "__main__":
    parser = ArgumentParser(description="Transcribe with tesseract model")
    parser.add_argument(
        "model_name",
        help="Name of tesseract model (must be in list of available models with 'tesseract --list-langs')",
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="The directory containing images to be transcribed",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="The output directory to store predicted transcriptions",
    )
    parser.add_argument(
        "--find_gt",
        action="store_true",
        help="If flagged, will try to read ground truth data from image_dir/txt",
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

    args.output_dir.mkdir(exist_ok=True, parents=True)

    df = transcribe(model_name=args.model_name, image_dir=args.image_dir)

    if args.find_gt:
        df = find_gt(df=df, image_dir=args.image_dir)

    output_csv = (
        args.output_dir / f"{args.image_dir.name}_{args.model_name}_predictions.csv"
    )

    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote predicted transcriptions to {output_csv}")
