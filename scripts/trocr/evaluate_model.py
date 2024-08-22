import json
from pathlib import Path

import datasets
import mlflow
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import samisk_ocr.trocr
from samisk_ocr.trocr.data_processing import transform_data

config = samisk_ocr.trocr.config.Config()
mlflow.set_tracking_uri(config.mlflow_url)
mlflow.set_experiment("TrOCR trocr-base-stage1 finetuning on samisk-v3-line")

# Load the samisk dataset from Huggingface Hub
ds = datasets.load_dataset("Teklia/samisk-v3-line")

# Specify what model we want to load
run_name = "bright-calf-956"
checkpoint_dir = Path(f"data/checkpoints/train_trocr_base_stage1/{run_name}/")
checkpoint_name = "final_model"
run_id = mlflow.search_runs(filter_string=f"run_name = '{run_name}'")["run_id"].item()

# Load the training and validation sets
train_set = ds["train"]
validation_set = ds["validation"]

# load processor and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = TrOCRProcessor.from_pretrained(checkpoint_dir / "processor")
model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir / checkpoint_name).to(device)


# Figure out the maximum token length in the training set, which we use this to control the maximum
# length of generated sequences. Specifically, we set the maximum length of generated sequences to
# 1.5 times the maximum token length in the training set (rounded down), which can lead to significant
# speedups during inference and evaluation.
tokens = processor.tokenizer(train_set["text"], padding="do_not_pad", max_length=128)
max_tokens = max(len(token) for token in tokens["input_ids"])
max_target_length = int(1.5 * max_tokens)


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


# Estimate the text for the validation set
def estimate_text(batch):
    transformed = transform_data(batch, processor=processor, max_target_length=max_target_length)
    tokens = model.generate(transformed["pixel_values"].to(device))
    return {"estimated_text": processor.batch_decode(tokens, batch_decode=True)}


validation_set_with_estimates = validation_set.map(estimate_text, batched=True, batch_size=32)

# Save the estimated text for the validation set to MLFlow
output = [
    {"text": row["text"], "estimated_text": row["estimated_text"]}
    for row in validation_set_with_estimates
]

with mlflow.start_run(run_id=run_id) as run:
    mlflow.log_text(
        json.dumps(output),
        f"evaluation_results/{checkpoint_name}/validation_estimates.json",
    )
