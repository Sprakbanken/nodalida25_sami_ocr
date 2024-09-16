from pathlib import Path

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    TRANSKRIBUS_EXPORT_DIR: Path = (
        Path(__file__).parent.parent / "data/transkribus_exports/train_data/train"
    )
    LANGUAGE_TSV_PARENT: Path = Path(__file__).parent.parent / "data"
