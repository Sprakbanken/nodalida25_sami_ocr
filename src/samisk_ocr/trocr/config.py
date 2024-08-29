from pathlib import Path

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    DATA_PATH: Path | None = None
    MLFLOW_HOST: str = "localhost"
    MLFLOW_PORT: int = 5431
    MLFLOW_PROTOCOL: str = "http"

    MLFLOW_ARTIFACT_RUN_INFO_DIR: Path = Path("run_info")
    MLFLOW_ARTIFACT_IMAGE_DIR: Path = Path("images")
    MLFLOW_ARTIFACT_PREDICTIONS_DIR: Path = Path("predictions")

    @property
    def mlflow_url(self) -> str:
        server = self.MLFLOW_HOST.removesuffix("/")
        url = f"{self.MLFLOW_PROTOCOL}://{server}:{self.MLFLOW_PORT}"
        return url

    class Config:
        env_file = ".env"
