import re
from functools import partial
from typing import Any

import datasets


def check_width_height_ratio(row: dict[str, Any], min_with_height_ratio: float) -> bool:
    width = row["xmax"] - row["xmin"]
    height = row["ymax"] - row["ymin"]
    return width > min_with_height_ratio * height


def preprocess_dataset(
    dataset: datasets.Dataset,
    min_len: int,
    min_with_height_ratio: float,
    include_page_30: bool,
    include_gt_pix: bool,
) -> datasets.Dataset:
    dataset = dataset.filter(lambda x: x["text_len"] >= min_len).filter(
        partial(check_width_height_ratio, min_with_height_ratio=min_with_height_ratio)
    )

    if not include_page_30:
        dataset = dataset.filter(lambda x: not x["page_30"])
    if not include_gt_pix:
        dataset = dataset.filter(lambda x: not x["gt_pix"])

    return dataset
