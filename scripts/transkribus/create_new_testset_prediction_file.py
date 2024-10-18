"""Fordi testsettet er fordelt på to transkribuseksporter lager jeg prediction-fila her"""

from pathlib import Path

import pandas as pd

from samisk_ocr.transkribus.export_to_prediction_file import get_line_transcriptions

if __name__ == "__main__":
    ### Change these as needed ###
    testset_p = Path("data/new_testset_with_newspapers")
    model_name = "transk_smi_nor_pred"

    testset_predictions1 = Path(
        "data/transkribus_exports/predictions/test_set/Testsett_Samisk_OCR_testing"
    )
    testset_predictions2 = Path("data/transkribus_exports/predictions/test_set_aviser")
    output_dir = Path("output/testset_preds/line_level")

    ##############################

    pred_df1 = get_line_transcriptions(testset_predictions1)
    pred_df1 = pred_df1[pred_df1.image.apply(lambda x: "pliktmonografi" not in x)]
    pred_df1.index = range(len(pred_df1))

    pred_df2 = get_line_transcriptions(testset_predictions2)
    pred_df = pd.concat([pred_df1, pred_df2], ignore_index=True)

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
