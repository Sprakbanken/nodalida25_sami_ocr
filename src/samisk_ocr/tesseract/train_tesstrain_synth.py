import json
import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset

from samisk_ocr.tesseract.train_tesstrain import (
    copy_data,
    copy_image_files_and_create_gt_files,
    create_train_eval_lists,
)
from samisk_ocr.utils import setup_logging

logger = logging.getLogger(__name__)


def create_tesstrain_data(
    path_to_synthetic_dataset: str,
    path_to_dataset: str,
    model_data_dir: Path,
    model_traindata_dir: Path,
):
    logger.info("Loading synthetic dataset...")
    train_dataset = load_dataset("imagefolder", data_dir=path_to_synthetic_dataset, split="train")

    logger.info("Loading validation dataset...")
    val_dataset = load_dataset("imagefolder", data_dir=path_to_dataset, split="validation")

    logger.info(f"Moving train data into {model_traindata_dir}...")
    copy_image_files_and_create_gt_files(
        train_dataset, output_dir=model_traindata_dir, id_col="unique_id"
    )

    logger.info(f"Moving validation data into {model_traindata_dir}...")
    copy_image_files_and_create_gt_files(
        val_dataset,
        output_dir=model_traindata_dir,
    )

    logger.info("Creating .lstmf and .box files")
    subprocess.run(["make", "-C", "tesstrain", "lists", f"MODEL_NAME={model_data_dir.name}"])

    logger.info("Creating custom list.train and list.eval files")
    create_train_eval_lists(
        train_ds=train_dataset,
        val_ds=val_dataset,
        output_dir=model_data_dir,
        prefix=f"data/{model_traindata_dir.name}/",
        train_id_col="unique_id",
    )


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Create training data and training script to train tesseract model"
    )
    parser.add_argument("model_name", help="Name of tesseract model")
    parser.add_argument("--tessdata", help="Path to tessdata (see tesseract_howto)", required=True)
    parser.add_argument("--tesstrain_repo", help="Path to tesstrain repo", default="tesstrain")
    parser.add_argument(
        "--synth_dataset_path", help="Path to synthetic dataset", default="data/syntetisk_dataset"
    )
    parser.add_argument("--dataset_path", help="Path to dataset", default="data/samisk_ocr_dataset")

    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument(
        "--start_model", type=str, help="Tesseract start model to continue training from"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If flagged, will plot checkpoints with tesstrain plot function after training",
    )
    parser.add_argument(
        "--copy_data",
        type=str,
        metavar="model",
        help="Will copy tesstrain-data from model instead of creating new",
        default="",
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
    setup_logging(source_script="train_tesstrain", log_level=args.log_level)
    logger.info(args)

    model_traindata_dir = Path(args.tesstrain_repo) / "data" / f"{args.model_name}-ground-truth/"
    model_data_dir = Path(args.tesstrain_repo) / "data" / args.model_name

    if args.copy_data:
        other_model_traindata_dir = (
            Path(args.tesstrain_repo) / "data" / f"{args.copy_data}-ground-truth/"
        )
        other_model_data_dir = Path(args.tesstrain_repo) / "data" / args.copy_data
        if not (other_model_traindata_dir.exists() and other_model_data_dir.exists()):
            logger.error(f"copy_data is flagged but {args.copy_data} model data files do not exist")
            exit()

        model_traindata_dir.mkdir()
        model_data_dir.mkdir()
        copy_data(
            model_data_dir, model_traindata_dir, other_model_traindata_dir, other_model_data_dir
        )
    else:
        model_traindata_dir.mkdir()
        model_data_dir.mkdir()
        create_tesstrain_data(
            path_to_synthetic_dataset=args.synth_dataset_path,
            path_to_dataset=args.dataset_path,
            model_data_dir=model_data_dir,
            model_traindata_dir=model_traindata_dir,
        )

    tesstrain_arg_list = [
        "make",
        "-C",
        args.tesstrain_repo,
        "training",
        f"MODEL_NAME={args.model_name}",
        f"EPOCHS={args.num_epochs}",
        f"LEARNING_RATE={args.learning_rate}",
        f"TESSDATA={args.tessdata}",
    ]

    if args.start_model:
        tesstrain_arg_list.append(f"START_MODEL={args.start_model}")

    model_stem = "".join([char for char in args.model_name if not char.isnumeric()])
    tesseract_models_model_dir = Path(f"tesseract_models/{model_stem}")
    tesseract_models_model_dir.mkdir(exist_ok=True)

    with (tesseract_models_model_dir / f"{args.model_name}_training_args.json").open("w+") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    subprocess.run(tesstrain_arg_list)
    if args.plot:
        subprocess.run(["make", "-C", args.tesstrain_repo, "plot", f"MODEL_NAME={args.model_name}"])
