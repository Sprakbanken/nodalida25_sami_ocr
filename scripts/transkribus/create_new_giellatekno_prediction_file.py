"""Fordi testsettet er fordelt på to transkribuseksporter lager jeg prediction-fila her"""

import argparse
import re
from pathlib import Path

import pandas as pd

from samisk_ocr.transkribus.export_to_prediction_file import get_line_transcriptions

if __name__ == "__main__":
    ### Change these as needed ###
    parser = argparse.ArgumentParser()
    parser.add_argument("testset_path", type=Path)
    parser.add_argument("predictions_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    testset_p = args.testset_path  # Path("data/giellatekno/huggingface")
    model_name = "transk_smi_nor_pred"

    testset_predictions = (
        args.predictions_path
    )  # Path("data/transkribus_exports/predictions/export_job_12356934_nor_smi")
    output_dir = args.output_dir  # Path("output/giellatekno_nor_sme_preds/line_level")

    ##############################

    pred_df = get_line_transcriptions(testset_predictions)
    pred_df = pred_df.assign(
        image=pred_df["image"].map(
            lambda s: re.sub(r"_\d{3}_\d{4}_\d{4}_\d{4}_\d{4}[.]jpg", ".png", s)
        ),
    )
    dataset_df = pd.read_csv(testset_p / "test" / "_metadata.csv")

    df = dataset_df.merge(
        pred_df,
        left_on="file_name",
        right_on="image",
        validate="1:1",
    )

    assert len(df) == len(dataset_df)

    output_csv = output_dir / f"{model_name}_test.csv"
    df.to_csv(output_csv, index=False)
