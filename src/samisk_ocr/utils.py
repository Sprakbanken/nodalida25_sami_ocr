import logging
import sys
from collections import namedtuple
from datetime import datetime

import pandas as pd

Bbox = namedtuple("Bbox", ["xmin", "ymin", "xmax", "ymax"])


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

    # Some images don't have numeric page numbers, but pandas likes columns of same type
    return pre, -1


def image_stem_to_urn_line_bbox(image_stem: str) -> tuple[str, int, Bbox]:
    """With page number as part of urn"""

    bbox = Bbox(*[int(e) for e in image_stem[-19:].split("_")])

    urn, line = image_stem[:-20].rsplit("_", maxsplit=1)
    return (urn, int(line), bbox)


def image_stem_to_urn_page_line_bbox(image_stem: str) -> tuple[str, int, int, Bbox]:
    try:
        bbox = Bbox(*[int(e) for e in image_stem[-19:].split("_")])
    except Exception:
        print(image_stem)
        return
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
