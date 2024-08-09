import logging
from datetime import datetime
import pandas as pd


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
