from __future__ import annotations

import itertools
from dataclasses import dataclass
from operator import attrgetter
from typing import TYPE_CHECKING, Iterable, TypedDict

import evaluate
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import transformers
from tqdm import tqdm
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


class BaseImageSaverCallback(TrainerCallback):
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

    def get_evaluated_examples(
        self, model: transformers.modeling_utils.PreTrainedModel
    ) -> list[EvaluatedExample]:
        raise NotImplementedError

    def get_filename(self, state: transformers.trainer_callback.TrainerState) -> str:
        raise NotImplementedError

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


class RandomImageSaverCallback(BaseImageSaverCallback):
    def __init__(
        self,
        processor: transformers.processing_utils.ProcessorMixin,
        validation_data: datasets.arrow_dataset.Dataset,
        processed_validation_data: datasets.arrow_dataset.Dataset,
        device: torch.device,
        artifact_image_dir: Path,
        save_frequency: int = 10,
    ):
        super().__init__(
            processor=processor,
            validation_data=validation_data,
            processed_validation_data=processed_validation_data,
            device=device,
            artifact_image_dir=artifact_image_dir,
            save_frequency=save_frequency,
        )

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

        # TODO: Lag en liste med alle evaluated examples bilder som vi så sorterer og henter ut rett antall fra
        return [
            EvaluatedExample(
                cer=cer_metric.compute(predictions=[pred_text], references=[true_text]),
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


class WorstImageSaverCallback(BaseImageSaverCallback):
    def __init__(
        self,
        processor: transformers.processing_utils.ProcessorMixin,
        validation_data: datasets.arrow_dataset.Dataset,
        processed_validation_data: datasets.arrow_dataset.Dataset,
        device: torch.device,
        artifact_image_dir: Path,
        save_frequency: int = 10,
        batch_size: int = 8,
    ):
        super().__init__(
            processor=processor,
            validation_data=validation_data,
            processed_validation_data=processed_validation_data,
            device=device,
            artifact_image_dir=artifact_image_dir,
            save_frequency=save_frequency,
        )
        self.batch_size = batch_size

    def get_evaluated_examples(
        self, model: transformers.modeling_utils.PreTrainedModel
    ) -> list[EvaluatedExample]:
        # Get the images and texts from the validation dataset
        images = self.validation_data["image"]
        texts = self.validation_data["text"]

        pred_texts = []
        processed_val_iterable = tqdm(self.processed_validation_data, desc="Predicting for images")
        for b in itertools.batched(processed_val_iterable, self.batch_size):
            pixel_values = torch.stack([bi["pixel_values"] for bi in b]).to(self.device)

            predictions = model.generate(pixel_values).tolist()
            batch_pred_texts = self.processor.batch_decode(predictions, skip_special_tokens=True)
            pred_texts.extend(batch_pred_texts)

        return sorted(
            [
                EvaluatedExample(
                    cer=cer_metric.compute(predictions=[pred_text], references=[true_text]),
                    image=px,
                    text=true_text,
                    estimated_text=pred_text,
                )
                for px, true_text, pred_text in zip(images, texts, pred_texts, strict=True)
            ],
            key=attrgetter("cer"),
            reverse=True,
        )[:5]

    def get_filename(self, state: transformers.trainer_callback.TrainerState) -> str:
        return (
            self.artifact_image_dir
            / "worst_lines"
            / f"predictions_step_{state.global_step:08d}.png"
        )
