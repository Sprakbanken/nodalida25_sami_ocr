pdm run python -m samisk_ocr.tesseract.transcribe sb_smi --output_dir output/valset_preds
pdm run python -m samisk_ocr.tesseract.transcribe sb_smi_nor_pred --output_dir output/valset_preds
pdm run python -m samisk_ocr.tesseract.transcribe smi --output_dir output/valset_preds
pdm run python -m samisk_ocr.tesseract.transcribe smi_nor --output_dir output/valset_preds
pdm run python -m samisk_ocr.tesseract.transcribe smi_nor_pred --output_dir output/valset_preds
pdm run python -m samisk_ocr.tesseract.transcribe smi_pred --output_dir output/valset_preds
pdm run python -m samisk_ocr.tesseract.transcribe synth_base --output_dir output/valset_preds
pdm run python -m samisk_ocr.tesseract.transcribe ub_smi --output_dir output/valset_preds