from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import mlflow

if TYPE_CHECKING:
    from pathlib import Path

    from samisk_ocr.trocr.config import Config


def log_file(run: mlflow.entities.Run, file_path: str, mlflow_artifact_dir: Path) -> None:
    """Log a file to MLflow as an artifact."""
    with open(file_path, "r") as file:
        content = file.read()
    mlflow.log_text(content, mlflow_artifact_dir / file_path.name, run_id=run.info.run_id)


def log_installed_packages(run: mlflow.entities.Run, mlflow_artifact_dir: Path) -> None:
    """Log installed packages to MLflow."""
    packages = subprocess.run(
        ["pip", "freeze"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    mlflow.log_text(
        packages.stdout, mlflow_artifact_dir / "requirements.txt", run_id=run.info.run_id
    )


def log_git_info(run: mlflow.entities.Run, mlflow_artifact_dir: Path) -> None:
    """Logs the current hash and the git status to MLflow as artifacts."""
    hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    mlflow.log_text(hash.stdout, mlflow_artifact_dir / "git_hash.txt", run_id=run.info.run_id)

    status = subprocess.run(
        ["git", "status"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    mlflow.log_text(status.stdout, mlflow_artifact_dir / "git_status.txt", run_id=run.info.run_id)


def log_config(run: mlflow.entities.Run, config: Config, mlflow_artifact_dir: Path) -> None:
    """Log the configuration to MLflow as an artifact."""
    mlflow.log_text(
        config.model_dump_json(indent=2),
        mlflow_artifact_dir / "config.json",
        run_id=run.info.run_id,
    )
