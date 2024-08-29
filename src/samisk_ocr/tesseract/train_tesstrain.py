import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path

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


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Create training data and training script to train tesseract model"
    )
    parser.add_argument("model_name", help="Name of tesseract model")
    parser.add_argument("--tessdata", help="Path to tessdata (see tesseract_howto)")
    parser.add_argument(
        "--tesstrain_repo", type=Path, help="Path to tesstrain repo", default="tesstrain"
    )
    parser.add_argument(
        "--dataset_path", help="Path to dataset", default="data/samisk_ocr_line_level_dataset"
    )
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

    logger.info("Loading dataset...")
    dataset = load_dataset("imagefolder", data_dir=args.dataset_path)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    if args.filter_wh:
        train_dataset = train_dataset.filter(lambda x: x["width"] > x["height"])
    if args.filter_len:
        train_dataset = train_dataset.filter(lambda x: x["text_len"] > args.filter_len)
    if args.page_30 == "exclude":
        train_dataset = train_dataset.filter(lambda x: not x["page_30"])
    elif args.page_30 == "only":
        train_dataset = train_dataset.filter(lambda x: x["page_30"])
    if args.gt_pix == "exclude":
        train_dataset = train_dataset.filter(lambda x: not x["gt_pix"])
    elif args.page_30 == "only":
        train_dataset = train_dataset.filter(lambda x: x["gt_pix"])

    model_traindata_dir = args.tesstrain_repo / "data" / f"{args.model_name}-ground-truth/"

    if model_traindata_dir.exists():
        if (
            len(list(model_traindata_dir.glob("*.txt")))
            != train_dataset.num_rows + val_dataset.num_rows
        ):
            logger.error(
                f"{model_traindata_dir} already exists, but number of files do not correspond to current dataset filters.\
                    Exiting"
            )
            exit()
    else:
        model_traindata_dir.mkdir()
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
    subprocess.run(["make", "-C", "tesstrain", "lists", f"MODEL_NAME={args.model_name}"])

    model_data_dir = args.tesstrain_repo / "data" / args.model_name
    model_data_dir.mkdir(exist_ok=True)

    logger.info("Creating custom list.train and list.eval files")
    create_train_eval_lists(
        train_ds=train_dataset,
        val_ds=val_dataset,
        output_dir=model_data_dir,
        prefix=f"data/{args.model_name}-ground-truth/",
    )

    del dataset, train_dataset, val_dataset

    tesstrain_arg_list = [
        "make",
        "-C",
        "tesstrain",
        "training",
        f"MODEL_NAME={args.model_name}",
        f"EPOCHS={args.num_epochs}",
        f"LEARNING_RATE={args.learning_rate}",
        f"TESSDATA={args.tessdata}",
    ]

    if args.start_model:
        tesstrain_arg_list.append(f"START_MODEL={args.start_model}")

    subprocess.run(tesstrain_arg_list)
