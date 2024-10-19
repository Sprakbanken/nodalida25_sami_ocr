# Get transcriptions on (new) testset for the best models of each type
# Make sure tesseract model is in tessdata directory
export TESSDATA="/usr/local/share/tessdata/"
#cp tesseract_models/sb_smi_nor_pred/sb_smi_nor_pred.traineddata $TESSDATA

DATASET="data/giellatekno/huggingface"
OUTPUT="output/giellatekno_nor_sme_preds"
TRANSKRIBUS_EXPORT_PATH="data/transkribus_exports/predictions/export_job_12356934_nor_smi"
pdm run python -m samisk_ocr.tesseract.transcribe sb_smi_nor_pred --dataset $DATASET --split test --output_dir $OUTPUT
pdm run python -m samisk_ocr.trocr.transcribe Sprakbanken/trocr_smi_pred_synth --dataset $DATASET --split test --output_dir $OUTPUT
pdm run python scripts/transkribus/create_new_giellatekno_prediction_file.py $DATASET $TRANSKRIBUS_EXPORT_PATH $OUTPUT/line_level
echo Adding langcodes to predictions
for file in $(find ${OUTPUT}/ | grep .csv)
do
    pdm run python -c "import pandas as pd; pd.read_csv('$file').assign(langcodes='[\'sme\']').to_csv('$file', index=False)"
done



DATASET="data/giellatekno/friis-huggingface_with_page"
OUTPUT="output/giellatekno_sme_friis_preds"
TRANSKRIBUS_EXPORT_PATH="data/transkribus_exports/predictions/export_job_12357443_smi_friis"
pdm run python scripts/transkribus/create_new_giellatekno_prediction_file.py $DATASET $TRANSKRIBUS_EXPORT_PATH $OUTPUT/line_level
pdm run python -m samisk_ocr.trocr.transcribe Sprakbanken/trocr_smi_pred_synth --dataset $DATASET --split test --output_dir $OUTPUT
pdm run python -m samisk_ocr.tesseract.transcribe sb_smi_nor_pred --dataset $DATASET --split test --output_dir $OUTPUT
echo Adding langcodes to predictions
for file in $(find ${OUTPUT}/ | grep .csv)
do
    pdm run python -c "import pandas as pd; pd.read_csv('$file').assign(langcodes='[\'sme\']').to_csv('$file', index=False)"
done