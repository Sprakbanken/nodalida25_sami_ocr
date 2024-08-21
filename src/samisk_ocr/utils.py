import logging
from datetime import datetime
import pandas as pd

from collections import namedtuple

Bbox = namedtuple("Bbox", ["xmin", "ymin", "xmax", "ymax"])


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
    urn_, line = image_stem[:-20].rsplit("_", maxsplit=1)
    urn, page = urn_.rsplit("_", maxsplit=1)
    return (urn, page, int(line), bbox)


def setup_logging(source_script: str, log_level: str):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Set up logging
    log_filename = f"logs/{source_script}_{current_time}.log"
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )


def clean_transcriptions(transcriptions: pd.Series) -> pd.Series:
    """Convert to string and strip"""
    return transcriptions.apply(str).apply(str.strip)
