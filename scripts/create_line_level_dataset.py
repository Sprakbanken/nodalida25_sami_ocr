import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree

import pandas as pd
from datasets import load_dataset

from samisk_ocr.transkribus_export_to_line_data import transkribus_export_to_lines
from samisk_ocr.utils import (
    image_stem_to_urn_page_line_bbox,
    page_image_stem_to_urn_page,
    setup_logging,
)

logger = logging.getLogger(__name__)


def write_urns_to_languages():
    """Write mapping from urn to language code for each urn in the train data"""
    doc_id_to_lang_df = pd.read_csv("data/trainset_languages.tsv", sep="\t")
    urns_to_langcodes = {}

    for e in Path("data/transkribus_exports/train_data/train").iterdir():
        for sub_dir in e.iterdir():
            df_ = doc_id_to_lang_df[doc_id_to_lang_df.dokument == sub_dir.name]
            langcodes = [e.språkkode for e in df_.itertuples()]
            urns = [
                page_image_stem_to_urn_page(path.stem)[0]
                for path in sub_dir.iterdir()
                if not (path.is_dir() or path.suffix in (".xml", ".json"))
            ]
            for urn in urns:
                urns_to_langcodes[urn] = langcodes

    for e in Path("data/transkribus_exports/train_data/GT_pix").iterdir():
        if not e.suffix == ".tif":
            continue
        urn = page_image_stem_to_urn_page(e.stem)[0]
        urns_to_langcodes[urn] = ["nor"]

    for e in Path("data/transkribus_exports/train_data/side_30").iterdir():
        if not e.is_dir():
            continue
        for sub_dir in e.iterdir():
            _, langcode = sub_dir.name.split("_")
            urns = [
                page_image_stem_to_urn_page(path.stem)[0]
                for path in sub_dir.iterdir()
                if not (path.is_dir() or path.suffix in (".xml", ".json"))
            ]
            for urn in urns:
                urns_to_langcodes[urn] = [langcode]

    testdata_page_urn_to_lang_df = pd.read_csv("data/testset_languages.tsv", sep="\t")
    urns = testdata_page_urn_to_lang_df.side_filnavn.apply(lambda x: x[:-5])
    for urn, langcode in zip(urns, testdata_page_urn_to_lang_df.språkkode):
        urns_to_langcodes[urn] = [langcode]

    with open("data/urns_to_langcodes.json", "w+") as f:
        json.dump(urns_to_langcodes, f, ensure_ascii=False, indent=4)


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
            sum_so_far += urn_lines[i][1]
            urns.append(urn_lines[i][0])
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
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "text_len",
        "page_30",
        "gt_pix",
    ]

    return metadata_df[columns_in_order]


def rearrange_train_and_val_files(
    metadata_df: pd.DataFrame, val_urns: dict[str, list[str]], dataset_dir: Path
) -> pd.DataFrame:
    """Create file train and val file structure, and return metadata with updated paths"""

    # Set from_path relative to (outer) dataset directory
    metadata_df["from_path"] = metadata_df.file_name.apply(
        lambda file_name: dataset_dir / "temp_train" / file_name
    )
    assert all(metadata_df.from_path.apply(lambda x: x.exists()))

    val_urn_list = [urn for urn_list in val_urns.values() for urn in urn_list]
    val_df = metadata_df[metadata_df.urn.isin(val_urn_list)]
    train_df = metadata_df[~metadata_df.urn.isin(val_urn_list)]

    train_dir = dataset_dir / "train"
    train_dir.mkdir(exist_ok=True)

    for e in train_df.itertuples():
        e.from_path.rename(train_dir / Path(e.file_name).name)

    train_df["file_name"] = train_df.file_name.apply(
        lambda file_name: "train/" + Path(file_name).name
    )

    val_dir = dataset_dir / "val"
    val_dir.mkdir(exist_ok=True)

    for e in val_df.itertuples():
        e.from_path.rename(val_dir / Path(e.file_name).name)

    val_df["file_name"] = val_df.file_name.apply(lambda file_name: "val/" + Path(file_name).name)

    metadata_df = pd.concat((train_df, val_df))
    metadata_df.index = range(len(metadata_df))
    metadata_df = metadata_df.drop(columns=["from_path"])
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

    # Create line level images from manually transcribed testdata
    temp_train = dataset_dir / "temp_train"
    if not temp_train.exists():
        temp_train.mkdir(parents=True)
        transkribus_export_to_lines(
            base_image_dir=Path("data/transkribus_exports/train_data/train"), output_dir=temp_train
        )
    train_metadata_df = encance_metadata_df(pd.read_csv(temp_train / "metadata.csv"))

    # Create valiation split
    validation_urns = create_val_split(train_metadata_df)
    metadata_df = rearrange_train_and_val_files(
        metadata_df=train_metadata_df, val_urns=validation_urns, dataset_dir=dataset_dir
    )

    # Add automatically transcribed page_30 files to train directory
    page_30 = dataset_dir / "train" / "side_30"
    if not page_30.exists():
        page_30.mkdir()
        transkribus_export_to_lines(
            base_image_dir=Path("data/transkribus_exports/train_data/side_30"),
            output_dir=page_30,
        )
    page_30_metadata_csv_path = page_30 / "metadata.csv"
    page_30_metadata_df = pd.read_csv(page_30_metadata_csv_path)
    page_30_metadata_df["page_30"] = [True] * len(page_30_metadata_df)
    page_30_metadata_df["file_name"] = page_30_metadata_df.file_name.apply(
        lambda file_name: "train/side_30/" + file_name
    )
    page_30_metadata_df = encance_metadata_df(page_30_metadata_df)
    metadata_df = pd.concat((metadata_df, page_30_metadata_df))

    # Add GTpix files to train directory
    gt_pix = dataset_dir / "train" / "GT_pix"
    if not gt_pix.exists():
        gt_pix.mkdir()
        transkribus_export_to_lines(
            base_image_dir=Path("data/transkribus_exports/train_data/GT_pix"),
            output_dir=gt_pix,
        )
    gt_pix_metadata_csv_path = gt_pix / "metadata.csv"
    gt_pix_metadata_df = pd.read_csv(gt_pix_metadata_csv_path)
    gt_pix_metadata_df["gt_pix"] = [True] * len(gt_pix_metadata_df)
    gt_pix_metadata_df["file_name"] = gt_pix_metadata_df.file_name.apply(
        lambda file_name: "train/GT_pix/" + file_name
    )
    gt_pix_metadata_df = encance_metadata_df(gt_pix_metadata_df)
    metadata_df = pd.concat((metadata_df, gt_pix_metadata_df))

    # Create line level images from test data
    test_dir = dataset_dir / "test"
    test_dir.mkdir()
    transkribus_export_to_lines(
        base_image_dir=Path("data/transkribus_exports/test_data"), output_dir=test_dir
    )
    test_metadata_csv_path = test_dir / "metadata.csv"
    test_metadata_df = encance_metadata_df(pd.read_csv(test_metadata_csv_path))
    test_metadata_df["file_name"] = test_metadata_df.file_name.apply(lambda x: "test/" + x)
    metadata_df = pd.concat((metadata_df, test_metadata_df))

    metadata_df.to_csv(dataset_dir / "metadata.csv", index=False)

    # Cleanup files
    test_metadata_csv_path.unlink()
    page_30_metadata_csv_path.unlink()
    gt_pix_metadata_csv_path.unlink()
    rmtree(temp_train)

    dataset = load_dataset("imagefolder", data_dir=dataset_dir)
    logger.info(f"Successfully created 🤗️ image dataset at {dataset_dir}")
