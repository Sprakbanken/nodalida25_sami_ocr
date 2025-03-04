from functools import partial
from math import ceil
from pathlib import Path

import mlflow
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
from samisk_ocr.mlflow.callbacks import (
    ImageSaverCallback,
)
from samisk_ocr.trocr.data_processing import transform_data
from samisk_ocr.trocr.dataset import load_dataset, preprocess_dataset

config = samisk_ocr.trocr.config.Config()
mlflow.set_tracking_uri(config.mlflow_url)
mlflow.set_experiment("TrOCR trocr-base-stage1 finetuning")
dataset_path = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the datasets
train_set = preprocess_dataset(
    load_dataset(
        config.DATA_PATH,
        split_path=Path("data/urns.json"),
        split="train",
        only_curated=True,
    ),
    min_len=3,
    min_with_height_ratio=2,
    include_page_30=False,
)
validation_set = preprocess_dataset(
    load_dataset(
        config.DATA_PATH,
        split_path=Path("data/urns.json"),
        split="val",
        only_curated=True,
    ),
    min_len=3,
    min_with_height_ratio=2,
    include_page_30=False,
)

# Load the TrOCR processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1").to(device)

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
processed_train_set = train_set.with_transform(transform_data_partial)
processed_validation_set = validation_set.with_transform(transform_data_partial)

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


with mlflow.start_run() as run:
    # Setup checkpoint dir
    experiment_name = Path(__file__).stem
    checkpoint_dir = Path(f"data/checkpoints/{experiment_name}/{run.info.run_name}/")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Log extra info for reproducibility
    samisk_ocr.mlflow.logging.log_git_info(run, config.MLFLOW_ARTIFACT_RUN_INFO_DIR)
    samisk_ocr.mlflow.logging.log_installed_packages(run, config.MLFLOW_ARTIFACT_RUN_INFO_DIR)
    samisk_ocr.mlflow.logging.log_file(run, Path(__file__), config.MLFLOW_ARTIFACT_RUN_INFO_DIR)

    # Setup trainer args
    batch_size = 8
    steps_per_epoch = ceil(len(processed_train_set) / batch_size)
    eval_steps = 5 * steps_per_epoch
    batched_eval_frequency = 2000
    training_args = Seq2SeqTrainingArguments(
        #
        # Training paramters
        fp16=False,
        learning_rate=1e-5,
        num_train_epochs=30,
        per_device_train_batch_size=8,
        remove_unused_columns=False,
        lr_scheduler_type=transformers.SchedulerType.COSINE_WITH_RESTARTS,
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
        save_strategy="epoch",  # Must be same as eval_strategy
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
            ImageSaverCallback(
                processor=processor,
                validation_data=validation_set,
                processed_validation_data=processed_validation_set,
                device=device,
                save_frequency=batched_eval_frequency,
                artifact_image_dir=config.MLFLOW_ARTIFACT_IMAGE_DIR,
            ),
        ],
    )

    processor.save_pretrained(checkpoint_dir / "processor")
    trainer.train()
    trainer.save_model(checkpoint_dir / "final_model")
