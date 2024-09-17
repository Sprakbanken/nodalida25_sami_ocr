import json
import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy2
from typing import Literal

from datasets import Dataset, load_dataset

from samisk_ocr.utils import setup_logging

logger = logging.getLogger(__name__)


def copy_image_files_and_create_gt_files(ds: Dataset, output_dir: Path):
    for i, e in enumerate(ds):
        file_stem = f"{e['urn']}_{i}"
        out_p = output_dir / f"{file_stem}.gt.txt"
        with out_p.open("w+") as f:
            f.write(e["text"])
        img = e["image"]
        img.save(output_dir / f"{file_stem}.png")


def create_train_eval_lists(train_ds: Dataset, val_ds: Dataset, output_dir: Path, prefix: str):
    train_lstmf_names = [f"{prefix}{e['urn']}_{i}.lstmf" for i, e in enumerate(train_ds)]
    with open(output_dir / "list.train", "w+") as f:
        f.write("\n".join(train_lstmf_names))

    val_lstmf_names = [f"{prefix}{e['urn']}_{i}.lstmf" for i, e in enumerate(val_ds)]
    with open(output_dir / "list.eval", "w+") as f:
        f.write("\n".join(val_lstmf_names))

    with open(output_dir / "all-lstmf", "w+") as f:
        f.write("\n".join(train_lstmf_names + val_lstmf_names))


def create_tesstrain_data(
    path_to_dataset: Path,
    model_data_dir: Path,
    model_traindata_dir: Path,
    filter_wh: bool,
    filter_len: int,
    page_30: Literal["exclude", "include", "only"],
    gt_pix: Literal["exclude", "include", "only"],
):
    logger.info("Loading dataset...")
    dataset = load_dataset("imagefolder", data_dir=path_to_dataset)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    if filter_wh:
        train_dataset = train_dataset.filter(lambda x: x["width"] > x["height"])
    if filter_len:
        train_dataset = train_dataset.filter(lambda x: x["text_len"] > filter_len)
    if page_30 == "exclude":
        train_dataset = train_dataset.filter(lambda x: not x["page_30"])
    elif page_30 == "only":
        train_dataset = train_dataset.filter(lambda x: x["page_30"])
    if gt_pix == "exclude":
        train_dataset = train_dataset.filter(lambda x: not x["gt_pix"])
    elif gt_pix == "only":
        train_dataset = train_dataset.filter(lambda x: x["gt_pix"])

    logger.info(f"Moving train data into {model_traindata_dir}...")
    copy_image_files_and_create_gt_files(
        train_dataset,
        output_dir=model_traindata_dir,
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
    )


def copy_data(
    model_data_dir: Path,
    model_traindata_dir: Path,
    other_model_traindata_dir: Path,
    other_model_data_dir: Path,
):
    # Copy training data
    for e in other_model_traindata_dir.iterdir():
        copy2(src=e, dst=model_traindata_dir / e.name)

    # Copy lstmf-list files
    eval_list = other_model_data_dir / "list.eval"
    train_list = other_model_data_dir / "list.train"
    all_list = other_model_data_dir / "all-lstmf"

    def replace_data_dir_filename(lstmf_filename: str) -> str:
        data_dir, other_model_traindata_dir_name, filename = lstmf_filename.split("/")
        return "/".join((data_dir, model_traindata_dir.name, filename))

    for file_ in (eval_list, train_list, all_list):
        filenames = file_.read_text().split("\n")
        new_filenames = [replace_data_dir_filename(e) for e in filenames]
        with open(model_data_dir / file_.name, "w+") as f:
            f.write("\n".join(new_filenames))


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Create training data and training script to train tesseract model"
    )
    parser.add_argument("model_name", help="Name of tesseract model")
    parser.add_argument("--tessdata", help="Path to tessdata (see tesseract_howto)")
    parser.add_argument("--tesstrain_repo", help="Path to tesstrain repo", default="tesstrain")
    parser.add_argument("--dataset_path", help="Path to dataset", default="data/samisk_ocr_dataset")
    parser.add_argument(
        "--filter_wh",
        action="store_true",
        help="If flagged, will filter out images where width < height",
    )
    parser.add_argument(
        "--filter_len",
        type=int,
        metavar="n",
        help="If provided, will filter out images where transcription is shorter than n",
    )
    parser.add_argument(
        "--page_30",
        choices=["only", "include", "exclude"],
        default="exclude",
        help="Whether to include, exclude or transfer only the page_30 data",
    )
    parser.add_argument(
        "--gt_pix",
        choices=["only", "include", "exclude"],
        default="exclude",
        help="Whether to include, exclude or transfer only the gt_pix data",
    )
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
    model_traindata_dir.mkdir()
    model_data_dir.mkdir()

    if args.copy_data:
        other_model_traindata_dir = (
            Path(args.tesstrain_repo) / "data" / f"{args.copy_data}-ground-truth/"
        )
        other_model_data_dir = Path(args.tesstrain_repo) / "data" / args.copy_data
        if not (other_model_traindata_dir.exists() and other_model_data_dir.exists()):
            logger.error(f"copy_data is flagged but {args.copy_data} model data files do not exist")
            exit()
        copy_data(
            model_data_dir, model_traindata_dir, other_model_traindata_dir, other_model_data_dir
        )
    else:
        create_tesstrain_data(
            path_to_dataset=args.dataset_path,
            model_data_dir=model_data_dir,
            model_traindata_dir=model_traindata_dir,
            filter_len=args.filter_len,
            filter_wh=args.filter_wh,
            page_30=args.page_30,
            gt_pix=args.gt_pix,
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
