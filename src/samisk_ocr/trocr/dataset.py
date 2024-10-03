import re
from functools import partial
from typing import Any

import datasets


def should_include_dataset_row(
    row: dict[str, Any],
    *,
    min_len: int,
    filter_width: bool,
    include_page_30: bool,
    include_gt_pix: bool,
    min_len_page_30: int,
) -> bool:
    if row["text_len"] < min_len:
        return False
    if filter_width and row["width"] <= row["height"]:
        return False
    if row["gt_pix"] and not include_gt_pix:
        return False
    if row["page_30"] and not include_page_30:
        return False
    if row["page_30"] and row["text_len"] < min_len_page_30:
        return False

    return True


def preprocess_dataset(
    dataset: datasets.Dataset,
    min_len: int,
    filter_width: bool,
    include_page_30: bool,
    include_gt_pix: bool,
    min_len_page_30: int,
) -> datasets.Dataset:
    dataset = dataset.filter(
        partial(
            should_include_dataset_row,
            min_len=min_len,
            filter_width=filter_width,
            include_page_30=include_page_30,
            include_gt_pix=include_gt_pix,
            min_len_page_30=min_len_page_30,
        )
    )

    return dataset
