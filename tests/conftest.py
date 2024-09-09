import shutil
from pathlib import Path

import pytest


@pytest.fixture
def setup_and_teardown_test_dirs():
    print("\nSetup: Setup testdata")
    ids = ["2893103", "2751369"]

    transkribus_exports_in = (
        Path(__file__).parent.parent / "data/transkribus_exports/train_data/train"
    )
    testdata_dir = Path(__file__).parent / "_test_data"
    transkribus_exports_out = testdata_dir / "transkribus_exports"
    temp_dir = testdata_dir / "temp_dir"
    dataset_dir = Path(__file__).parent / "_test_data/dataset_dir"

    transkribus_exports_out.mkdir(parents=True)
    temp_dir.mkdir()
    dataset_dir.mkdir()

    for e in ["test_data", "train_data/GT_pix", "train_data/side_30", "train_data/train"]:
        export_out = transkribus_exports_out / e / ids[0]

        export_data = transkribus_exports_in / ids[0]
        shutil.copytree(src=export_data, dst=export_out)

    # add one more example to train in order to create val split
    export_out = transkribus_exports_out / "train_data/train" / ids[1]
    export_data = transkribus_exports_in / ids[1]
    shutil.copytree(src=export_data, dst=export_out)

    yield
    print("\nTeardown: Removing testdata")
    shutil.rmtree(testdata_dir)


@pytest.fixture
def transkribus_export_dir():
    return Path(__file__).parent / "_test_data/transkribus_exports"


@pytest.fixture
def dataset_dir():
    return Path(__file__).parent / "_test_data/dataset_dir"


@pytest.fixture
def temp_dir():
    return Path(__file__).parent / "_test_data/temp_dir"
