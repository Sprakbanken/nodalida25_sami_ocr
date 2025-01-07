# Get transcriptions on (new) testset for the best models of each type
# Make sure tesseract model is in tessdata directory
export TESSDATA="/usr/local/share/tessdata/"
#cp tesseract_models/sb_smi_nor_pred/sb_smi_nor_pred.traineddata $TESSDATA

DATASET="data/giellatekno/huggingface_nor_sme-fixed"
OUTPUT="output/giellatekno_nor_sme_preds-fixed"
TRANSKRIBUS_EXPORT_PATH="data/transkribus_exports/4672812"
pdm run python -m samisk_ocr.tesseract.transcribe sb_smi_nor_pred --dataset $DATASET --split test --output_dir $OUTPUT
pdm run python -m samisk_ocr.trocr.transcribe Sprakbanken/trocr_smi_pred_synth --dataset $DATASET --split test --output_dir $OUTPUT
pdm run python scripts/transkribus/create_new_giellatekno_prediction_file.py $DATASET $TRANSKRIBUS_EXPORT_PATH $OUTPUT/line_level
echo Adding langcodes to predictions
for file in $(find ${OUTPUT}/ | grep .csv)
do
    pdm run python -c "import pandas as pd; pd.read_csv('$file').assign(langcodes='[\'sme\']').to_csv('$file', index=False)"
done
