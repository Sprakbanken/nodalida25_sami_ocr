pdm run python -m evaluate_predictions output/predictions/line_level/test_fin_sme_predictions.csv data/line_level/test --line --model_name fin_sme --base_model_language fin &
pdm run python -m evaluate_predictions output/predictions/line_level/test_nor_predictions.csv data/line_level/test --line --model_name nor --base_model_language nor &
pdm run python -m evaluate_predictions output/predictions/line_level/test_nor_sme_predictions.csv data/line_level/test --line --model_name nor_sme --base_model_language nor &
pdm run python -m evaluate_predictions output/predictions/line_level/test_nor_smx_200_predictions.csv data/line_level/test --line --model_name nor_smx_200 --base_model_language nor &
pdm run python -m evaluate_predictions output/predictions/line_level/test_nor_smx_201_predictions.csv data/line_level/test --line --model_name nor_smx_201 --base_model_language nor &
pdm run python -m evaluate_predictions output/predictions/line_level/test_nor_smx_202_predictions.csv data/line_level/test --line --model_name nor_smx_202 --base_model_language nor &
pdm run python -m evaluate_predictions output/predictions/line_level/test_nor_smx_205_predictions.csv data/line_level/test --line --model_name nor_smx_205 --base_model_language nor &
pdm run python -m evaluate_predictions output/predictions/line_level/test_nor_smx_206_predictions.csv data/line_level/test --line --model_name nor_smx_206 --base_model_language nor &
pdm run python -m evaluate_predictions output/predictions/line_level/test_nor_smx_207_predictions.csv data/line_level/test --line --model_name nor_smx_207 --base_model_language nor &
pdm run python -m evaluate_predictions output/predictions/line_level/test_nor_smx_30000_predictions.csv data/line_level/test --line --model_name nor_smx_30000 --base_model_language nor &
pdm run python -m evaluate_predictions output/predictions/line_level/test_nor_smx_train_rotate_remove_predictions.csv data/line_level/test --line --model_name nor_smx_train_rotate_remove --base_model_language nor &
pdm run python -m evaluate_predictions output/predictions/line_level/test_sme_friis_predictions.csv data/line_level/test --line --model_name sme_friis &
pdm run python -m evaluate_predictions output/predictions/line_level/test_sme_predictions.csv data/line_level/test --line --model_name sme &
pdm run python -m evaluate_predictions output/predictions/line_level/test_smx2_30000_predictions.csv data/line_level/test --line --model_name smx2_30000 &
pdm run python -m evaluate_predictions output/predictions/line_level/transkribus_med_base_predictions.csv data/line_level/test --line --model_name transkribus_med_base --map_transkribus --base_model_language fin &
pdm run python -m evaluate_predictions output/predictions/line_level/transkribus_uten_base_predictions.csv data/line_level/test --line --model_name transkribus_uten_base --map_transkribus &




pdm run python -m samisk_ocr.tesseract_transcribe fin --line &
pdm run python -m samisk_ocr.tesseract_transcribe fin_sme --line &
pdm run python -m samisk_ocr.tesseract_transcribe nor --line &
pdm run python -m samisk_ocr.tesseract_transcribe nor_sme --line &
pdm run python -m samisk_ocr.tesseract_transcribe sme --line &
pdm run python -m samisk_ocr.tesseract_transcribe sme_friis --line &
