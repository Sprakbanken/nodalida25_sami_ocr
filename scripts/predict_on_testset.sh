# Get transcriptions on (new) testset for the best models of each type
# Make sure tesseract model is in tessdata directory
export TESSDATA="/usr/local/share/tessdata/"
cp tesseract_models/sb_smi_nor_pred/sb_smi_nor_pred.traineddata $TESSDATA

pdm run python scripts/baseline/predict_baseline.py --output_dir output/testset_preds & 
pdm run python -m samisk_ocr.tesseract.transcribe sb_smi_nor_pred --dataset data/new_testset_with_newspapers --split test --output_dir output/testset_preds &
pdm run python -m samisk_ocr.trocr.transcribe Sprakbanken/trocr_smi_pred_synth --dataset data/new_testset_with_newspapers --split test --output_dir output/testset_preds &
pdm run python scripts/transkribus/create_new_testset_prediction_file.py
