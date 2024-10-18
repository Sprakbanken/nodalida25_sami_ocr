# Evaluate predictions
pdm run python -m samisk_ocr.evaluate_predictions output/testset_preds/line_level --output_dir output/testset_evaluation --remove_pliktmono --log_level DEBUG
echo "See evaluation results at output/testset_evaluation"