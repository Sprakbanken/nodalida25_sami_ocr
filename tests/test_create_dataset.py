import logging
from pathlib import Path

import pandas as pd
import pytest
from datasets import load_dataset

from samisk_ocr.create_line_level_dataset import Args, create_dataset
from samisk_ocr.utils import setup_logging

from .utils import Config

logger = logging.getLogger(__name__)
setup_logging(source_script="test_create_dataset", log_level="DEBUG")


@pytest.mark.skipif(
    not Config().TRANSKRIBUS_EXPORT_DIR.exists(), reason="TRANSKRIBUS_EXPORT_DIR does not exist"
)
@pytest.mark.usefixtures("setup_and_teardown_test_dirs")
def test_create_line_level_dataset(temp_dir: Path, transkribus_export_dir: Path, dataset_dir: Path):
    assert len(list(dataset_dir.iterdir())) == 0
    args = Args(
        dataset_dir=dataset_dir,
        temp_dir=temp_dir,
        transkribus_export_dir=transkribus_export_dir,
        log_level="",
    )
    create_dataset(args=args)
    assert len(list(dataset_dir.iterdir())) > 0

    df_test = pd.read_csv(dataset_dir / "test" / "_metadata.csv")
    df_train = pd.read_csv(dataset_dir / "train" / "_metadata.csv")
    df_val = pd.read_csv(dataset_dir / "val" / "_metadata.csv")

    dataset = load_dataset(str(dataset_dir))
    assert dataset["train"].num_rows == len(df_train)
    assert dataset["validation"].num_rows == len(df_val)
    assert dataset["test"].num_rows == len(df_test)


if __name__ == "__main__":
    test_create_line_level_dataset()
