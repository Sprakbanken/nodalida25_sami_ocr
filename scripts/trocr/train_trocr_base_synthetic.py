import logging
import random
from functools import partial
from math import ceil
from pathlib import Path

import datasets
import mlflow
import numpy as np
import torch
import transformers
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

import samisk_ocr.trocr
from samisk_ocr.metrics import compute_cer, compute_wer
from samisk_ocr.mlflow.callbacks import (
    BatchedMultipleEvaluatorsCallback,
    ConcatCEREvaluator,
    ConcatWEREvaluator,
    MetricSummaryEvaluator,
    MultipleEvaluatorsCallback,
    RandomImageSaverCallback,
    WorstTranscriptionImageEvaluator,
)
from samisk_ocr.trocr.data_processing import DatasetSampler, transform_data
from samisk_ocr.trocr.dataset import preprocess_dataset
from samisk_ocr.utils import setup_logging

logger = logging.getLogger(__name__)
setup_logging(source_script=Path(__file__).stem, log_level="INFO")

config = samisk_ocr.trocr.config.Config()
print(config.mlflow_url)
mlflow.set_tracking_uri(config.mlflow_url)
mlflow.set_experiment("TrOCR trocr-base-printed finetuning-synthetic")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the datasets
validation_set = preprocess_dataset(
    datasets.load_dataset("imagefolder", data_dir=config.DATA_PATH, split="validation"),
    min_len=1,
    filter_width=True,
    include_page_30=False,
    include_gt_pix=False,
    min_len_page_30=5,
)
synth_dataset = datasets.load_dataset("imagefolder", data_dir=config.SYNTH_DATA_PATH)
logger.info("Loading validation data")
synth_validation_set = synth_dataset["validation"].shuffle(seed=42).take(1000)
logger.info("Loading training data")
train_set = synth_dataset["train"]
logger.info("Data loaded")

# Load the TrOCR processor and model
logger.info("Loading TrOCR processor and model")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)

logger.info("Setting model and processor params")
# Figure out the maximum token length in the training set, which we use this to control the maximum
# length of generated sequences. Specifically, we set the maximum length of generated sequences to
# 1.5 times the maximum token length in the training set (rounded down), which can lead to significant
# speedups during inference and evaluation.
tokens = processor.tokenizer(train_set["text"], padding="do_not_pad", max_length=128)
max_tokens = max(len(token) for token in tokens["input_ids"])
max_target_length = int(1.5 * max_tokens)


# Set the data transform and apply it to our training and validation sets
transform_data_partial = partial(
    transform_data, processor=processor, max_target_length=max_target_length
)
processed_train_set = train_set.with_transform(transform_data_partial, output_all_columns=True)
processed_validation_set = validation_set.with_transform(
    transform_data_partial, output_all_columns=True
)
processed_synth_validation_set = synth_validation_set.with_transform(
    transform_data_partial, output_all_columns=True
)

# Ensure that the processor and model use the same special tokens
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# Make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# Set the beam search parameters used for inference.
model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
model.generation_config.max_length = max_target_length
model.generation_config.early_stopping = False
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.length_penalty = 2.0
model.generation_config.num_beams = 4

# Set which metrics to log:
evaluators = [
    MetricSummaryEvaluator(compute_cer, np.mean, "mean_cer"),
    MetricSummaryEvaluator(compute_wer, np.mean, "mean_wer"),
    MetricSummaryEvaluator(compute_cer, np.median, "median_cer"),
    MetricSummaryEvaluator(compute_wer, np.median, "median_wer"),
    *[
        MetricSummaryEvaluator(compute_cer, partial(np.percentile, q=q), f"{q}percentile_cer")
        for q in [95, 90, 75, 25]
    ],
    *[
        MetricSummaryEvaluator(compute_wer, partial(np.percentile, q=q), f"{q}percentile_wer")
        for q in [95, 90, 75, 25]
    ],
    ConcatCEREvaluator(),
    ConcatWEREvaluator(),
    WorstTranscriptionImageEvaluator(
        key="worst_cer_images", artifact_dir=config.MLFLOW_ARTIFACT_IMAGE_DIR
    ),
]


with mlflow.start_run() as run:
    logger.info("Starting run %s", run.info.run_name)

    # Setup checkpoint dir
    experiment_name = Path(__file__).stem
    checkpoint_dir = Path(f"data/checkpoints/{experiment_name}/{run.info.run_name}/")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Log extra info for reproducibility
    samisk_ocr.mlflow.logging.log_git_info(run, config.MLFLOW_ARTIFACT_RUN_INFO_DIR)
    samisk_ocr.mlflow.logging.log_installed_packages(run, config.MLFLOW_ARTIFACT_RUN_INFO_DIR)
    samisk_ocr.mlflow.logging.log_file(run, Path(__file__), config.MLFLOW_ARTIFACT_RUN_INFO_DIR)
    samisk_ocr.mlflow.logging.log_config(run, config, config.MLFLOW_ARTIFACT_RUN_INFO_DIR)

    # Setup trainer args
    batch_size = 8
    steps_per_epoch = ceil(len(processed_train_set) / batch_size)
    eval_steps = steps_per_epoch // 10
    batched_eval_frequency = eval_steps // 4
    training_args = Seq2SeqTrainingArguments(
        #
        # Training paramters
        fp16=False,
        learning_rate=1e-6,
        num_train_epochs=100,
        per_device_train_batch_size=8,
        remove_unused_columns=False,
        #
        # Evaluation parameters
        eval_strategy="steps",
        eval_steps=eval_steps,
        greater_is_better=False,
        logging_steps=50,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        #
        # Checkpoint parameters
        load_best_model_at_end=True,
        metric_for_best_model="eval_cer",
        output_dir=checkpoint_dir,
        save_strategy="steps",  # Must be same as eval_strategy
        save_steps=eval_steps,
    )

    # Setup trainer
    eval_func = partial(samisk_ocr.trocr.metrics.compute_metrics, processor=processor)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=eval_func,
        train_dataset=processed_train_set,
        eval_dataset=processed_validation_set,
        data_collator=default_data_collator,
        callbacks=[
            RandomImageSaverCallback(
                processor=processor,
                validation_data=validation_set,
                processed_validation_data=processed_validation_set,
                device=device,
                save_frequency=batched_eval_frequency,
                artifact_image_dir=config.MLFLOW_ARTIFACT_IMAGE_DIR,
            ),
            MultipleEvaluatorsCallback(
                evaluators=evaluators,
                processor=processor,
                validation_data=validation_set,
                processed_validation_data=processed_validation_set,
                frequency=eval_steps,
                key_prefix="eval_",
                artifact_path=config.MLFLOW_ARTIFACT_PREDICTIONS_DIR / "real",
            ),
            MultipleEvaluatorsCallback(
                evaluators=evaluators,
                processor=processor,
                validation_data=synth_validation_set,
                processed_validation_data=processed_synth_validation_set,
                frequency=eval_steps,
                key_prefix="eval_synth_",
                artifact_path=config.MLFLOW_ARTIFACT_PREDICTIONS_DIR / "synth",
                unique_identifiers=("unique_id",)
            ),
            BatchedMultipleEvaluatorsCallback(
                evaluators=evaluators,
                processor=processor,
                batch_sampler=DatasetSampler(
                    dataset=validation_set,
                    processed_dataset=processed_validation_set,
                    batch_size=16,
                ),
                frequency=batched_eval_frequency,
                key_prefix="batched_eval_",
            ),
            BatchedMultipleEvaluatorsCallback(
                evaluators=evaluators,
                processor=processor,
                batch_sampler=DatasetSampler(
                    dataset=synth_validation_set,
                    processed_dataset=processed_synth_validation_set,
                    batch_size=16,
                ),
                frequency=batched_eval_frequency,
                key_prefix="batched_synth_eval_",
            ),
            BatchedMultipleEvaluatorsCallback(
                evaluators=evaluators,
                processor=processor,
                batch_sampler=DatasetSampler(
                    dataset=train_set,
                    processed_dataset=processed_train_set,
                    batch_size=16,
                ),
                frequency=batched_eval_frequency,
                key_prefix="batched_train_",
            ),
        ],
    )

    logger.info("Saving processor")
    processor.save_pretrained(checkpoint_dir / "processor")
    logger.info("Starting training")
    trainer.train()
    logger.info("Training done, saving final model...")
    trainer.save_model(checkpoint_dir / "final_model")
