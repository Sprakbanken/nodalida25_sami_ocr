# Evaluate predictions
pdm run python -m samisk_ocr.evaluate_predictions \
    output/giellatekno_nor_sme_preds/line_level \
    --output_dir output/giellatekno_nor_sme_evaluation \
    --remove_pliktmono \
    --log_level DEBUG
echo "See evaluation results at output/giellatekno_nor_sme_evaluation"

pdm run python -m samisk_ocr.evaluate_predictions \
    output/giellatekno_sme_friis_preds/line_level \
    --output_dir output/giellatekno_sme_friis_evaluation \
    --remove_pliktmono \
    --log_level DEBUG
echo "See evaluation results at output/giellatekno_sme_friis_evaluation"