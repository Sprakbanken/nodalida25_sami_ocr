# Evaluate predictions
pdm run python -m \
    samisk_ocr.evaluate_predictions output/baseline_preds/line_level \
    --output_dir output/baseline_evaluation \
    --remove_pliktmono \
    --log_level DEBUG \
    --baseline_csv output/baseline_preds/line_level/baseline_test.csv
echo "See evaluation results at output/baseline_evaluation"