import shutil
from pathlib import Path

import pytest

from .utils import Config


@pytest.fixture
def setup_and_teardown_test_dirs(tmp_path):
    ids = ["2893103", "2706089"]

    transkribus_exports_in = Config().TRANSKRIBUS_EXPORT_DIR
    transkribus_exports_out = tmp_path / "transkribus_exports"
    temp_dir = tmp_path / "temp_dir"

    transkribus_exports_out.mkdir(parents=True)
    temp_dir.mkdir()

    for e in ["test_data", "train_data/GT_pix", "train_data/side_30", "train_data/train"]:
        export_out = transkribus_exports_out / e / ids[0]

        export_data = transkribus_exports_in / ids[0]
        shutil.copytree(src=export_data, dst=export_out)

    # add one more example to train in order to create val split
    export_out = transkribus_exports_out / "train_data/train" / ids[1]
    export_data = transkribus_exports_in / ids[1]
    shutil.copytree(src=export_data, dst=export_out)

    # rename side_30 inner directory (utils.get_urn_to_langcode_map expects filename to contain langcode separated with '_')
    side_30_inner_dir = next(
        (transkribus_exports_out / "train_data" / "side_30" / ids[0]).glob("*/")
    )
    side_30_inner_dir.replace(
        transkribus_exports_out / "train_data" / "side_30" / ids[0] / "sme_42"
    )


@pytest.fixture
def transkribus_export_dir(tmp_path: Path):
    out = tmp_path / "transkribus_exports"
    out.mkdir(exist_ok=True)
    return out


@pytest.fixture
def dataset_dir(tmp_path: Path):
    out = tmp_path / "dataset_dir"
    out.mkdir(exist_ok=True)
    return out


@pytest.fixture
def temp_dir(tmp_path: Path):
    out = tmp_path / "temp_dir"
    out.mkdir(exist_ok=True)
    return out
