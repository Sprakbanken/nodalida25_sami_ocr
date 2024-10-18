# TESSERACT models (see tesseract_models/README.md)
pdm run python -m samisk_ocr.tesseract.transcribe sb_smi --output_dir output/valset_preds
pdm run python -m samisk_ocr.tesseract.transcribe sb_smi_nor_pred --output_dir output/valset_preds
pdm run python -m samisk_ocr.tesseract.transcribe smi --output_dir output/valset_preds
pdm run python -m samisk_ocr.tesseract.transcribe smi_nor --output_dir output/valset_preds
pdm run python -m samisk_ocr.tesseract.transcribe smi_nor_pred --output_dir output/valset_preds
pdm run python -m samisk_ocr.tesseract.transcribe smi_pred --output_dir output/valset_preds
pdm run python -m samisk_ocr.tesseract.transcribe synth_base --output_dir output/valset_preds
pdm run python -m samisk_ocr.tesseract.transcribe ub_smi --output_dir output/valset_preds

# TRANSKRIBUS exports (see data/transkribus_exports/README.md)
pdm run python -m samisk_ocr.transkribus.export_to_prediction_file transk_smi data/transkribus_exports/predictions/val_set/our_line_level_layout_w_lm/smi --output_dir output/valset_preds
pdm run python -m samisk_ocr.transkribus.export_to_prediction_file transk_smi_nor data/transkribus_exports/predictions/val_set/our_line_level_layout_w_lm/smi_nor --output_dir output/valset_preds
pdm run python -m samisk_ocr.transkribus.export_to_prediction_file transk_smi_nor_pred data/transkribus_exports/predictions/val_set/our_line_level_layout_w_lm/smi_nor_pred --output_dir output/valset_preds
pdm run python -m samisk_ocr.transkribus.export_to_prediction_file transk_smi_pred data/transkribus_exports/predictions/val_set/our_line_level_layout_w_lm/smi_pred --output_dir output/valset_preds
pdm run python -m samisk_ocr.transkribus.export_to_prediction_file transk_smi_ub data/transkribus_exports/predictions/val_set/our_line_level_layout_w_lm/smi_ub --output_dir output/valset_preds

# TROCR
pdm run python -m samisk_ocr.trocr.transcribe Sprakbanken/trocr_smi_pred_synth --output_dir output/valset_preds
pdm run python -m samisk_ocr.trocr.transcribe Sprakbanken/trocr_smi_pred --output_dir output/valset_preds
pdm run python -m samisk_ocr.trocr.transcribe Sprakbanken/trocr_smi_synth --output_dir output/valset_preds
pdm run python -m samisk_ocr.trocr.transcribe Sprakbanken/trocr_smi_nor_pred --output_dir output/valset_preds
pdm run python -m samisk_ocr.trocr.transcribe Sprakbanken/trocr_smi_nor --output_dir output/valset_preds
pdm run python -m samisk_ocr.trocr.transcribe Sprakbanken/trocr_smi --output_dir output/valset_preds
# denne failer nok:
pdm run python -m samisk_ocr.trocr.transcribe Sprakbanken/trocr_smi_nor_pred_synth --output_dir output/valset_preds

