import json
from pathlib import Path
from typing import Literal

import datasets


def load_dataset(
    data_dir: str,
    split_path: Path,
    split: Literal["train", "test"] = "train",
    only_curated: bool = True,
) -> datasets.Dataset:
    urn_splits = json.loads(split_path.read_text())
    full_dataset = datasets.load_dataset("imagefolder", data_dir=data_dir, split="train")

    return full_dataset.filter(lambda x: x["urn"] in set(urn_splits[split]))


def preprocess_dataset(
    dataset: datasets.Dataset,
    min_len: int,
    min_with_height_ratio: float,
    include_page_30: bool = False,
) -> datasets.Dataset:
    dataset = dataset.filter(lambda x: x["text_len"] >= min_len).filter(
        lambda x: x["width"] > min_with_height_ratio * x["height"]
    )

    if not include_page_30:
        dataset = dataset.filter(lambda x: not x["page_30"])
    return dataset
