import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy2

import pandas as pd
from datasets import load_dataset

from samisk_ocr.transkribus_export_to_line_data import transkribus_export_to_lines
from samisk_ocr.utils import (
    image_stem_to_urn_page_line_bbox,
    page_image_stem_to_urn_page,
    setup_logging,
    write_urns_to_languages,
)

logger = logging.getLogger(__name__)


def create_val_split(metadata_df: pd.DataFrame) -> dict[str, list[str]]:
    # we dont have line level (or page level) language anntotations (only urn-level),
    # so the books with multiple languages just go to training
    single_language_df = metadata_df[metadata_df.langcodes.apply(len) == 1]
    single_language_df["langcode"] = single_language_df.langcodes.apply(lambda x: x[0])

    val_size = 0.25
    val_urns = {}
    for lang, df_ in single_language_df.groupby("langcode"):
        urn_lines = sorted(
            [(urn, len(df__)) for urn, df__ in df_.groupby("urn")], key=lambda x: x[1], reverse=True
        )
        val_num_target = int(len(df_) * val_size)
        urns = []
        sum_so_far = 0
        i = -1
        while sum_so_far < val_num_target:
            diff_to_target = val_num_target - sum_so_far
            next_urn, next_nr = urn_lines[i]
            # dont add next urn if total number of val lines goes further from target than it already is
            if (
                sum_so_far + next_nr > val_num_target
                and (sum_so_far + next_nr) - val_num_target > diff_to_target
            ):
                logger.debug(
                    f"For {lang}, number of lines in val set would have been {sum_so_far+next_nr}, breaking"
                )
                break
            sum_so_far += next_nr
            urns.append(next_urn)
            i -= 1
        logger.info(f"Number of lines in validation set for {lang}")
        logger.info(f"Target number of lines {val_num_target}")
        logger.info(f"Actual number of lines {sum_so_far}")
        val_urns[lang] = urns
    return val_urns


def encance_metadata_df(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Add more metadata to columns to metadata csv"""

    metadata_df["width"] = metadata_df.xmax - metadata_df.xmin
    metadata_df["height"] = metadata_df.ymax - metadata_df.ymin
    image_stems = metadata_df["file_name"].apply(lambda x: Path(x).stem)
    urns, pages, lines, _ = zip(*image_stems.apply(image_stem_to_urn_page_line_bbox))
    metadata_df["urn"] = urns
    metadata_df["page"] = pages
    metadata_df["line"] = lines
    metadata_df["text_len"] = metadata_df.text.apply(str).apply(len)

    with open("data/urns_to_langcodes.json") as f:
        urn_to_langcodes = json.load(f)
    metadata_df["langcodes"] = metadata_df.urn.apply(lambda x: urn_to_langcodes[x])

    if "page_30" not in metadata_df.columns:
        metadata_df["page_30"] = [False] * len(metadata_df)
    if "gt_pix" not in metadata_df.columns:
        metadata_df["gt_pix"] = [False] * len(metadata_df)

    columns_in_order = [
        "file_name",
        "text",
        "urn",
        "langcodes",
        "page",
        "line",
        "width",
        "height",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "text_len",
        "page_30",
        "gt_pix",
    ]

    return metadata_df[columns_in_order]


def copy_files(metadata_df: pd.DataFrame, from_dir: Path, to_dir: Path):
    # Set from_path relative to from directory
    from_paths = metadata_df.file_name.apply(lambda file_name: from_dir / file_name)
    assert all(from_paths.apply(lambda x: x.exists()))
    for file_name, from_path in zip(metadata_df.file_name, from_paths):
        copy2(src=from_path, dst=to_dir / Path(file_name).name)


def rearrange_train_and_val_files(
    metadata_df: pd.DataFrame,
    val_urns: dict[str, list[str]],
    temp_train_dir: Path,
    dataset_dir: Path,
) -> pd.DataFrame:
    """Create file train and val file structure, and return metadata with updated paths"""

    val_urn_list = [urn for urn_list in val_urns.values() for urn in urn_list]
    val_df = metadata_df[metadata_df.urn.isin(val_urn_list)]
    train_df = metadata_df[~metadata_df.urn.isin(val_urn_list)]

    train_dir = dataset_dir / "train"
    train_dir.mkdir(parents=True)

    copy_files(train_df, from_dir=temp_train_dir, to_dir=train_dir)
    train_df["file_name"] = train_df.file_name.apply(
        lambda file_name: "train/" + Path(file_name).name
    )

    val_dir = dataset_dir / "val"
    val_dir.mkdir()
    copy_files(val_df, from_dir=temp_train_dir, to_dir=val_dir)
    val_df["file_name"] = val_df.file_name.apply(lambda file_name: "val/" + Path(file_name).name)

    metadata_df = pd.concat((train_df, val_df))
    metadata_df.index = range(len(metadata_df))
    return metadata_df


if __name__ == "__main__":
    parser = ArgumentParser(description="Create a 🤗️ datasets image dataset from the data")
    parser.add_argument("dataset_dir", type=Path, help="Output dir to store dataset")
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    args = parser.parse_args()
    setup_logging(source_script="create_line_level_dataset", log_level=args.log_level)

    # Write mapping from urn to language code from dataset info files
    write_urns_to_languages()

    dataset_dir = args.dataset_dir
    temp_line_level_dir = dataset_dir.parent / f"{dataset_dir.name}_temp_line_level"

    # Create line level images from manually transcribed testdata
    temp_train = temp_line_level_dir / "train"
    if not temp_train.exists():
        temp_train.mkdir(parents=True)
        transkribus_export_to_lines(
            base_image_dir=Path("data/transkribus_exports/train_data/train"), output_dir=temp_train
        )
    train_metadata_df = encance_metadata_df(pd.read_csv(temp_train / "metadata.csv"))

    # Create valiation split
    validation_urns = create_val_split(train_metadata_df)
    metadata_df = rearrange_train_and_val_files(
        metadata_df=train_metadata_df,
        val_urns=validation_urns,
        dataset_dir=dataset_dir,
        temp_train_dir=temp_train,
    )

    # Add automatically transcribed page_30 files to train directory
    temp_page_30 = temp_line_level_dir / "side_30"
    if not temp_page_30.exists():
        temp_page_30.mkdir()
        transkribus_export_to_lines(
            base_image_dir=Path("data/transkribus_exports/train_data/side_30"),
            output_dir=temp_page_30,
        )
    page_30_metadata_df = pd.read_csv(temp_page_30 / "metadata.csv")
    page_30 = dataset_dir / "train" / "page_30"
    page_30.mkdir()

    copy_files(page_30_metadata_df, from_dir=temp_page_30, to_dir=page_30)

    page_30_metadata_df["page_30"] = [True] * len(page_30_metadata_df)

    page_30_metadata_df["file_name"] = page_30_metadata_df.file_name.apply(
        lambda file_name: "train/page_30/" + Path(file_name).name
    )
    page_30_metadata_df = encance_metadata_df(page_30_metadata_df)
    metadata_df = pd.concat((metadata_df, page_30_metadata_df))

    # Add GTpix files to train directory
    temp_gt_pix = temp_line_level_dir / "GT_pix"
    if not temp_gt_pix.exists():
        temp_gt_pix.mkdir()
        transkribus_export_to_lines(
            base_image_dir=Path("data/transkribus_exports/train_data/GT_pix"),
            output_dir=temp_gt_pix,
        )

    gt_pix_metadata_df = pd.read_csv(temp_gt_pix / "metadata.csv")
    gt_pix = dataset_dir / "train" / "GT_pix"
    gt_pix.mkdir()

    copy_files(gt_pix_metadata_df, from_dir=temp_gt_pix, to_dir=gt_pix)

    gt_pix_metadata_df["gt_pix"] = [True] * len(gt_pix_metadata_df)
    gt_pix_metadata_df["file_name"] = gt_pix_metadata_df.file_name.apply(
        lambda file_name: "train/GT_pix/" + Path(file_name).name
    )

    gt_pix_metadata_df = encance_metadata_df(gt_pix_metadata_df)
    metadata_df = pd.concat((metadata_df, gt_pix_metadata_df))

    # Create line level images from test data
    temp_test_dir = temp_line_level_dir / "test"
    if not temp_test_dir.exists():
        transkribus_export_to_lines(
            base_image_dir=Path("data/transkribus_exports/test_data"), output_dir=temp_test_dir
        )

    test_metadata_df = encance_metadata_df(pd.read_csv(temp_test_dir / "metadata.csv"))
    test_dir = dataset_dir / "test"
    test_dir.mkdir()

    copy_files(test_metadata_df, from_dir=temp_test_dir, to_dir=test_dir)
    test_metadata_df["file_name"] = test_metadata_df.file_name.apply(
        lambda file_name: "test/" + Path(file_name).name
    )

    metadata_df = pd.concat((metadata_df, test_metadata_df))
    metadata_df.to_csv(dataset_dir / "metadata.csv", index=False)

    dataset = load_dataset("imagefolder", data_dir=dataset_dir)
    logger.info(f"Successfully created 🤗️ image dataset at {dataset_dir}")