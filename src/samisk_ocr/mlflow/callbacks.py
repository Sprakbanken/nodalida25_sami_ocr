from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import evaluate
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import transformers
from transformers.trainer_callback import TrainerCallback

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    import accelerate
    import datasets
    import PIL.Image
    from typing_extensions import Unpack  # Python 3.12 can use Unpack for TypedDicts

    from samisk_ocr.trocr.types import TransformedData

cer_metric = evaluate.load("cer", trust_remote_code=True)


@dataclass
class EvaluatedExample:
    cer: float
    image: PIL.Image.Image
    text: str
    estimated_text: str

    def visualise(self, ax: plt.Axes) -> None:
        ax.imshow(self.image, cmap="gray")
        ax.set_title(
            f"     Text: {self.text}\n"
            f"Estimated: {self.estimated_text}\n"
            f"      CER: {self.cer}",
            loc="left",
            fontname="monospace",
        )
        ax.axis("off")


class CallbackKwargs(TypedDict):
    tokenizer: transformers.models.vit.image_processing_vit.ViTImageProcessor
    optimizer: accelerate.optimizer.AcceleratedOptimizer
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR
    train_dataloader: accelerate.data_loader.DataLoaderShard
    eval_dataloader: accelerate.data_loader.DataLoaderShard | None


class BatchEvalCallback(TrainerCallback):
    def __init__(
        self,
        compute_metrics: Callable[
            [transformers.trainer_utils.EvalPrediction], dict[str, float]
        ],
        batch_sampler: Iterator[TransformedData],
        eval_steps: int,
        prefix: str = "batched_eval",
    ) -> None:
        self.compute_metrics = compute_metrics
        self.batch_sampler = batch_sampler
        self.eval_steps = eval_steps
        self.prefix = prefix

    def on_step_end(
        self,
        args: transformers.training_args_seq2seq.Seq2SeqTrainingArguments,
        state: transformers.trainer_callback.TrainerState,
        control: transformers.trainer_callback.TrainerControl,
        model: transformers.modeling_utils.PreTrainedModel,
        **kwargs: Unpack[CallbackKwargs],
    ) -> None:
        if self.eval_steps is None or state.global_step % self.eval_steps != 0:
            return
        # Sample randomly from the dataset
        sample = next(self.batch_sampler)
        # Unpack sampled dataset and move to device before running inference
        pixel_values = sample["pixel_values"].to(model.device)
        labels = sample["labels"]
        prediction = model.generate(pixel_values).tolist()

        # Compute metrics
        prediction = transformers.trainer_utils.EvalPrediction(
            predictions=prediction, label_ids=labels.copy(), inputs=pixel_values
        )

        metrics_mean = {
            f"{self.prefix}_{k}": v for k, v in self.compute_metrics(prediction).items()
        }

        mlflow.log_metrics(metrics_mean, step=state.global_step)


class ImageSaverCallback(TrainerCallback):
    def __init__(
        self,
        processor: transformers.processing_utils.ProcessorMixin,
        validation_data: datasets.arrow_dataset.Dataset,
        processed_validation_data: datasets.arrow_dataset.Dataset,
        device: torch.device,
        artifact_image_dir: Path,
        save_frequency: int = 10,
    ):
        self.processor = processor
        self.validation_data = validation_data
        self.processed_validation_data = processed_validation_data
        self.device = device
        self.save_frequency = save_frequency
        self.artifact_image_dir = artifact_image_dir

        rng = np.random.default_rng(42)
        self.indices = [
            int(i) for i in rng.choice(len(self.validation_data), 5, replace=False)
        ]
        self.samples = [self.validation_data[i] for i in self.indices]
        self.processed_samples = [
            self.processed_validation_data[i] for i in self.indices
        ]
        self.pixel_values = torch.stack(
            [b["pixel_values"] for b in self.processed_samples]
        ).to(device)

        self.images = [b["image"] for b in self.samples]
        self.texts = [b["text"] for b in self.samples]

    def on_step_end(
        self,
        args: transformers.training_args_seq2seq.Seq2SeqTrainingArguments,
        state: transformers.trainer_callback.TrainerState,
        control: transformers.trainer_callback.TrainerControl,
        model: transformers.modeling_utils.PreTrainedModel,
        **kwargs: Unpack[CallbackKwargs],
    ) -> None:
        if self.save_frequency is None or state.global_step % self.save_frequency != 0:
            return

        batch = self.processed_samples
        # Unpack sampled dataset and move to device before running inference

        predictions = model.generate(self.pixel_values).tolist()
        decoded_preds = self.processor.batch_decode(
            predictions, skip_special_tokens=True
        )

        evaluated_examples = []
        for px, true_text, pred_text in zip(self.images, self.texts, decoded_preds):
            cer = cer_metric.compute(predictions=[pred_text], references=[true_text])
            evaluated_examples.append(
                EvaluatedExample(
                    cer=cer,
                    image=px,
                    text=true_text,
                    estimated_text=pred_text,
                )
            )
        # Create a with the five constant examples
        fig, axs = plt.subplots(5, 1, figsize=(10, 11), tight_layout=True)

        for ax, example in zip(axs, evaluated_examples):
            example.visualise(ax)

        mlflow.log_figure(
            fig,
            self.artifact_image_dir / f"predictions_step_{state.global_step:08d}.png",
        )
        plt.close(fig)
