from __future__ import annotations

import itertools
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence, TypedDict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers.trainer_callback import TrainerCallback

from samisk_ocr.metrics import compute_cer, compute_wer

if TYPE_CHECKING:
    import accelerate
    import datasets
    import PIL.Image
    from typing_extensions import Unpack  # type: ignore # False positive

    from samisk_ocr.metrics import Metric
    from samisk_ocr.mlflow.types import Evaluator, ReductionFunction
    from samisk_ocr.trocr.types import InputData, ProcessedData


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


class RandomImageSaverCallback(TrainerCallback):
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
        self.indices = [int(i) for i in rng.choice(len(self.validation_data), 5, replace=False)]
        self.samples = [self.validation_data[i] for i in self.indices]
        self.processed_samples = [self.processed_validation_data[i] for i in self.indices]
        self.pixel_values = torch.stack([b["pixel_values"] for b in self.processed_samples]).to(
            device
        )

        self.images = [b["image"] for b in self.samples]
        self.texts = [b["text"] for b in self.samples]

    def get_evaluated_examples(
        self, model: transformers.modeling_utils.PreTrainedModel
    ) -> list[EvaluatedExample]:
        # Unpack sampled dataset and move to device before running inference
        predictions = model.generate(
            self.pixel_values
        ).tolist()  # Få predictions til å kjøre på alle bilder
        pred_texts = self.processor.batch_decode(predictions, skip_special_tokens=True)

        return [
            EvaluatedExample(
                cer=compute_cer(
                    true_text,
                    pred_text,
                ),
                image=px,
                text=true_text,
                estimated_text=pred_text,
            )
            for px, true_text, pred_text in zip(self.images, self.texts, pred_texts)
        ]

    def get_filename(self, state: transformers.trainer_callback.TrainerState) -> str:
        return (
            self.artifact_image_dir
            / "random_lines"
            / f"predictions_step_{state.global_step:08d}.png"
        )

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

        evaluated_examples = self.get_evaluated_examples(model)
        fig = self.make_matplotlib_figure(evaluated_examples)

        mlflow.log_figure(fig, self.get_filename(state))
        plt.close(fig)

    def make_matplotlib_figure(self, evaluated_examples: EvaluatedExample) -> plt.Figure:
        # Create a with the five constant examples
        fig, axs = plt.subplots(
            len(evaluated_examples),
            1,
            figsize=(10, 2 * len(evaluated_examples) + 1),
            tight_layout=True,
        )

        for evaluated_example, ax in zip(evaluated_examples, axs):
            evaluated_example.visualise(ax)
        return fig


class MultipleEvaluatorsCallback(TrainerCallback):
    def __init__(
        self,
        evaluators: Sequence[Evaluator],
        processor: transformers.processing_utils.ProcessorMixin,
        validation_data: datasets.arrow_dataset.Dataset,
        processed_validation_data: datasets.arrow_dataset.Dataset,
        frequency: int,
        key_prefix: str,
        artifact_path: Path,
        batch_size: int = 8,
        unique_identifiers: Sequence[str] = ("urn", "page", "line"),
    ):
        self.evaluators = evaluators
        self.processor = processor
        self.processed_validation_data = processed_validation_data
        self.validation_data = validation_data
        self.frequency = frequency
        self.key_prefix = key_prefix
        self.batch_size = batch_size
        self.artifact_path = artifact_path
        self.unique_identifiers = unique_identifiers

    def on_step_end(
        self,
        args: transformers.training_args_seq2seq.Seq2SeqTrainingArguments,
        state: transformers.trainer_callback.TrainerState,
        control: transformers.trainer_callback.TrainerControl,
        model: transformers.modeling_utils.PreTrainedModel,
        **kwargs: Unpack[CallbackKwargs],
    ) -> None:
        if self.frequency is None or state.global_step % self.frequency != 0:
            return

        pred_texts = []
        processed_val_iterable = tqdm(self.processed_validation_data, desc="Predicting for images")
        for b in itertools.batched(processed_val_iterable, self.batch_size):
            pixel_values = torch.stack([bi["pixel_values"] for bi in b]).to(model.device)

            predictions = model.generate(pixel_values).tolist()
            batch_pred_texts = self.processor.batch_decode(predictions, skip_special_tokens=True)
            pred_texts.extend(batch_pred_texts)

        # Store the predictions as an artifact before we run the evaluators in case one of them crash
        mlflow.log_dict(
            {
                "predictions": pred_texts,
                "true": self.validation_data["text"],
            } | {
                identifier: self.validation_data[identifier] for identifier in self.unique_identifiers
            },
            self.artifact_path / f"{state.global_step:08d}.json",
        )

        for evaluator in self.evaluators:
            evaluator(self.validation_data, pred_texts, state.global_step, self.key_prefix)


class BatchedMultipleEvaluatorsCallback(TrainerCallback):
    def __init__(
        self,
        evaluators: Sequence[Evaluator],
        processor: transformers.processing_utils.ProcessorMixin,
        batch_sampler: Iterable[tuple[InputData, ProcessedData]],
        frequency: int,
        key_prefix: str,
    ):
        self.evaluators = evaluators
        self.processor = processor
        self.batch_sampler = batch_sampler
        self.frequency = frequency
        self.key_prefix = key_prefix

    def on_step_end(
        self,
        args: transformers.training_args_seq2seq.Seq2SeqTrainingArguments,
        state: transformers.trainer_callback.TrainerState,
        control: transformers.trainer_callback.TrainerControl,
        model: transformers.modeling_utils.PreTrainedModel,
        **kwargs: Unpack[CallbackKwargs],
    ) -> None:
        if self.frequency is None or state.global_step % self.frequency != 0:
            return

        # Sample randomly from the dataset
        sample, processed_sample = next(self.batch_sampler)
        # Unpack sampled dataset and move to device before running inference
        pixel_values = processed_sample["pixel_values"].to(model.device)
        predictions = model.generate(pixel_values).tolist()
        pred_texts = self.processor.batch_decode(predictions, skip_special_tokens=True)

        for evaluator in self.evaluators:
            evaluator(sample, pred_texts, state.global_step, self.key_prefix)


class MetricSummaryEvaluator:
    def __init__(self, metric: Metric, reduction_function: ReductionFunction, key: str) -> None:
        self.metric = metric
        self.reduction_function = reduction_function
        self.key = key

    def __call__(
        self,
        data: datasets.arrow_dataset.Dataset,
        pred_texts: list[str],
        step: int,
        key_prefix: str,
    ) -> None:
        metric_values = [
            self.metric(ground_truth=true, transcription=pred)
            for pred, true in zip(pred_texts, data["text"])
        ]
        summary = self.reduction_function(metric_values)
        mlflow.log_metric(f"{key_prefix}{self.key}", summary, step=step)


class WorstTranscriptionImageEvaluator:
    def __init__(
        self, key: str = "worst_cer_images", artifact_dir: Path = Path("artifacts")
    ) -> None:
        self.key = key
        self.artifact_dir = artifact_dir

    def __call__(
        self,
        data: datasets.arrow_dataset.Dataset,
        pred_texts: list[str],
        step: int,
        key_prefix: str,
    ) -> None:
        images = data["image"]
        texts = data["text"]

        examples = sorted(
            [
                EvaluatedExample(
                    cer=compute_cer(ground_truth=true_text, transcription=pred_text),
                    image=px,
                    text=true_text,
                    estimated_text=pred_text,
                )
                for px, true_text, pred_text in zip(images, texts, pred_texts)
            ],
            key=attrgetter("cer"),
            reverse=True,
        )[:5]

        fig, axs = plt.subplots(5, 1, figsize=(10, 10), tight_layout=True)
        for ax, example in zip(axs, examples):
            ax.imshow(example.image, cmap="gray")
            ax.set_title(
                f"     Text: {example.text}\n"
                f"Estimated: {example.estimated_text}\n"
                f"      CER: {example.cer}",
                loc="left",
                fontname="monospace",
            )
            ax.axis("off")

        file_name = self.artifact_dir / f"{key_prefix}{self.key}" / f"{step:08d}.png"
        mlflow.log_figure(fig, file_name)
        plt.close(fig)


class ConcatCEREvaluator:
    def __call__(
        self,
        data: datasets.arrow_dataset.Dataset,
        pred_texts: list[str],
        step: int,
        key_prefix: str,
    ) -> None:
        concat_true = "".join(data["text"])
        concat_pred = "".join(pred_texts)

        mlflow.log_metric(
            f"{key_prefix}concat_cer",
            compute_cer(ground_truth=concat_true, transcription=concat_pred),
            step=step,
        )


class ConcatWEREvaluator:
    def __call__(
        self,
        data: datasets.arrow_dataset.Dataset,
        pred_texts: list[str],
        step: int,
        key_prefix: str,
    ) -> None:
        concat_true = " ".join(data["text"])
        concat_pred = " ".join(pred_texts)

        mlflow.log_metric(
            f"{key_prefix}concat_wer",
            compute_wer(ground_truth=concat_true, transcription=concat_pred),
            step=step,
        )
