# Get transcriptions on (new) testset for the best models of each type
# Make sure tesseract model is in tessdata directory
export TESSDATA="/usr/local/share/tessdata/"
cp tesseract_models/sb_smi_nor_pred/sb_smi_nor_pred.traineddata $TESSDATA

pdm run python scripts/baseline/predict_baseline.py --output_dir output/baseline_preds & 
pdm run python -m samisk_ocr.tesseract.transcribe sb_smi_nor_pred --dataset data/baseline_huggingface --split test --output_dir output/baseline_preds &
pdm run python -m samisk_ocr.trocr.transcribe Sprakbanken/trocr_smi_pred_synth --dataset data/baseline_huggingface --split test --output_dir output/baseline_preds &

TRANSKRIBUS_EXPORT_PATH=data/transkribus_exports/baseline/4731412
pdm run python \
    scripts/transkribus/create_new_baseline_prediction_file.py \
    data/baseline_huggingface \
    $TRANSKRIBUS_EXPORT_PATH \
    output/baseline_preds/line_level
