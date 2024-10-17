pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/tess_fin_sme_val.csv --base_model_language fin &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/tess_fin_val.csv --base_model_language fin &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/tess_nor_sme_val.csv --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/tess_nor_val.csv --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/tess_sme_friis_val.csv &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/tess_sme_val.csv &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/tess_smi_nor_pred_val.csv --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/tess_smi_nor_val.csv --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/tess_smi_pred_val.csv --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/tess_smi_val.csv --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/tess_ub_smi_val.csv &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/transk_smi_nor_pred_val.csv --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/transk_smi_nor_val.csv --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/transk_smi_val.csv --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/transk_ub_smi_val.csv &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/trocr_smi_nor_val.csv --base_model_language eng &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/trocr_smi_val.csv --base_model_language eng &
