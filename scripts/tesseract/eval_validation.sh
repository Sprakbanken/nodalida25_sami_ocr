pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/fin_sme_val.csv --base_model_language fin &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/fin_val.csv --base_model_language fin &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/nor_sme_val.csv --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/nor_smi_nor_val.csv --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/nor_smi_val.csv --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/nor_val.csv --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/sme_friis_val.csv &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/sme_val.csv &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/smi_val.csv &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/tr_ocr_nor_smi_val.csv --base_model_language eng &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/tr_ocr_smi_val.csv --base_model_language eng &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/transkribus_med_alt_val.csv --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/transkribus_med_base_norsk_val.csv --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/transkribus_med_base_val.csv --base_model_language nor &
pdm run python -m samisk_ocr.evaluate_predictions output/predictions/line_level/transkribus_uten_base_val.csv --base_model_language nor &
