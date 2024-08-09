import pandas as pd
from pathlib import Path
import logging
from utils import setup_logging
from argparse import ArgumentParser

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def read_text_files(image_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    texts = []
    for e in df.itertuples():
        filename_stem = Path(e.image).stem
        gt_text_file = image_dir / "txt" / f"{filename_stem}.txt"
        if not gt_text_file.exists():
            logger.error(f"Could not find textfile for image {e.image}")
            exit()
        texts.append(gt_text_file.read_text())
    df["transcription"] = texts
    return df


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Create transcription file from transkribus export"
    )
    parser.add_argument(
        "model_name",
        help="Name of transkribus model",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="The directory containing image files and txt/ directory with transcription files",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="The output directory to store predicted transcriptions",
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

    images = [img.name for img in args.input_dir.glob("*.jpg")]

    df = pd.DataFrame({"image": images})
    df = read_text_files(df=df, image_dir=args.input_dir)

    output_csv = (
        args.output_dir / f"{args.input_dir.name}_{args.model_name}_predictions.csv"
    )

    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote predicted transcriptions to {output_csv}")
