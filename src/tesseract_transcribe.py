import pandas as pd
from pathlib import Path
from PIL import Image
import pytesseract
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from PIL import Image
import pytesseract
import logging
from utils import setup_logging
from argparse import ArgumentParser

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def transcribe_with_gt(
    model_name: str,
    ground_truth_dir: Path,
    image_file_ext: str = ".tif",
    text_file_ext: str = ".txt",
) -> pd.DataFrame:
    """Run tesseract model on images in ground_truth_dir, and read actual transcriptions from the textfiles in the same directory"""
    transcriptions = {
        "model_name": [],
        "image": [],
        "transcription": [],
        "ground_truth": [],
    }

    img_files = sorted(
        [e for e in ground_truth_dir.iterdir() if e.suffix == image_file_ext]
    )
    txt_files = sorted(
        [e for e in ground_truth_dir.iterdir() if e.suffix == text_file_ext]
    )

    for img_file, txt_file in tqdm(zip(img_files, txt_files), total=len(img_files)):
        transcription = pytesseract.image_to_string(
            Image.open(img_file), lang=model_name
        )
        if not transcription:
            logger.debug(f"No transcription for {img_file}")
            logger.debug(transcription)
        transcriptions["image"].append(img_file.name)
        transcriptions["model_name"].append(model_name)
        transcriptions["transcription"].append(transcription)
        transcriptions["ground_truth"].append(txt_file.read_text())

    df = pd.DataFrame(transcriptions)
    return df


def transcribe(model_name: str, image_dir: Path) -> pd.DataFrame:
    """Run tesseract model on all files in image_dir"""
    transcriptions = {"model_name": [], "image": [], "transcription": []}

    for img in image_dir.iterdir():
        transcription = pytesseract.image_to_string(Image.open(img), lang=model_name)
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
        "image_dir",
        type=Path,
        help="The directory containing images to be transcribed",
    )
    parser.add_argument("output_dir", type=Path, help="The output directory")
    parser.add_argument(
        "--gt",
        type=bool,
        default=True,
        help="If True, will look for ground truth transcriptions in image dir (if False, assumes image_dir only contains images)",
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

    if args.gt:
        df = transcribe_with_gt(
            model_name=args.model_name,
            ground_truth_dir=args.image_dir,
        )

        df.to_csv(
            args.output_dir
            / f"{args.image_dir.name}_{args.model_name}_transcriptions_gt.csv",
            index=False,
        )
    else:
        df = transcribe(model_name=args.model_name, image_dir=args.image_dir)
        df.to_csv(
            args.output_dir
            / f"{args.image_dir.name}_{args.model_name}_transcriptions.csv",
            index=False,
        )
