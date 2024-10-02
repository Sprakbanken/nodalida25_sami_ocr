pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/val_nor_smi_nor_predictions.csv --line --model_name nor_smi_nor --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/val_nor_smi_predictions.csv  --line --model_name nor_smi --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/val_smi_predictions.csv --line --model_name smi &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/val_fin_predictions.csv --line --model_name fin --base_model_language fin &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/val_fin_sme_predictions.csv --line --model_name fin_sme --base_model_language fin &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/val_nor_sme_predictions.csv --line --model_name nor_sme --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/val_sme_predictions.csv --line --model_name sme &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/val_sme_friis_predictions.csv --line --model_name sme_friis
