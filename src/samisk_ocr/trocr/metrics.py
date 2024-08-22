from __future__ import annotations

from typing import TYPE_CHECKING

import evaluate

from samisk_ocr.clean_text_data import clean

if TYPE_CHECKING:
    import transformers.models.trocr.processing_trocr
    import transformers.trainer_utils


cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


def compute_metrics(
    pred: transformers.trainer_utils.EvalPrediction,
    processor: transformers.models.trocr.processing_trocr.TrOCRProcessor,
) -> dict[str, float]:
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = [clean(s) for s in processor.batch_decode(pred_ids, skip_special_tokens=True)]
    label_str = [clean(s) for s in processor.batch_decode(labels_ids, skip_special_tokens=True)]

    out = {
        "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        "wer": wer_metric.compute(predictions=pred_str, references=label_str),
    }

    return out
