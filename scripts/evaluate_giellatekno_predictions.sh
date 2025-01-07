# Evaluate predictions
pdm run python -m samisk_ocr.evaluate_predictions \
    output/giellatekno_nor_sme_preds-fixed/line_level \
    --output_dir output/giellatekno_nor_sme_evaluation-fixed \
    --remove_pliktmono \
    --log_level DEBUG
echo "See evaluation results at output/giellatekno_nor_sme_evaluation-fixed"
