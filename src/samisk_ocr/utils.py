import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd


class Bbox(NamedTuple):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


def page_image_stem_to_urn_page(image_stem: str) -> tuple[str, int]:
    """Split page image filename stem into urn and page number"""
    # For books, the last part of the filename is the page number
    pre, post = image_stem.rsplit("_", maxsplit=1)
    if post.isnumeric():
        return (pre, int(post))

    # For newspapers, it is often the second to last part
    pre_pre, pre_post = pre.rsplit("_", maxsplit=1)

    # if the number is too large, its likely part of the urn and not a page number
    if pre_post.isnumeric() and len(pre_post) < 5:
        return (pre_pre, int(pre_post))

    # Some images don't have numeric page numbers
    return pre, pd.NA


def image_stem_to_urn_line_bbox(image_stem: str) -> tuple[str, int, Bbox]:
    """With page number as part of urn"""

    bbox = Bbox(*[int(e) for e in image_stem[-19:].split("_")])

    urn, line = image_stem[:-20].rsplit("_", maxsplit=1)
    return (urn, int(line), bbox)


def image_stem_to_urn_page_line_bbox(image_stem: str) -> tuple[str, int, int, Bbox]:
    bbox = Bbox(*[int(e) for e in image_stem[-19:].split("_")])
    urn_page, line = image_stem[:-20].rsplit("_", maxsplit=1)
    urn, page = page_image_stem_to_urn_page(urn_page)
    return (urn, page, int(line), bbox)


def setup_logging(source_script: str, log_level: str):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Set up logging
    log_filename = f"logs/{source_script}_{current_time}.log"
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler(stream=sys.stdout)],
    )


def clean_transcriptions(transcriptions: pd.Series) -> pd.Series:
    """Convert to string and strip"""
    return transcriptions.apply(str).apply(str.strip)


def get_urn_to_langcode_map(
    trainset_languages: Path = Path("data/trainset_languages.tsv"),
    testset_languages: Path = Path("data/testset_languages.tsv"),
    train_data_path: Path = Path("data/transkribus_exports/train_data/train"),
    gt_pix_path: Path = Path("data/transkribus_exports/train_data/GT_pix"),
    page_30_path: Path = Path("data/transkribus_exports/train_data/side_30"),
) -> dict[str, list[str]]:
    """Get mapping from urn to language code for each urn in the data"""
    doc_id_to_lang_df = pd.read_csv(trainset_languages, sep="\t")
    urns_to_langcodes = {}

    for e in train_data_path.iterdir():
        for sub_dir in e.iterdir():
            df_ = doc_id_to_lang_df[doc_id_to_lang_df.dokument == sub_dir.name]
            langcodes = df_.språkkode.to_list()
            urns = [page_image_stem_to_urn_page(path.stem)[0] for path in sub_dir.glob("*.jpg")]
            for urn in urns:
                urns_to_langcodes[urn] = langcodes

    for e in gt_pix_path.glob("*.tif"):
        urn = page_image_stem_to_urn_page(e.stem)[0]
        urns_to_langcodes[urn] = ["nor"]

    for e in page_30_path.glob("*/"):
        for sub_dir in e.iterdir():
            _, langcode = sub_dir.name.split("_")
            urns = [page_image_stem_to_urn_page(path.stem)[0] for path in sub_dir.glob("*.jpg")]
            for urn in urns:
                urns_to_langcodes[urn] = [langcode]

    testdata_page_urn_to_lang_df = pd.read_csv(testset_languages, sep="\t")
    urns = testdata_page_urn_to_lang_df.side_filnavn.apply(lambda x: x[:-5])
    for urn, langcode in zip(urns, testdata_page_urn_to_lang_df.språkkode):
        urns_to_langcodes[urn] = [langcode]

    return urns_to_langcodes
