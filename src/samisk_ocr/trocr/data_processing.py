from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from transformers.models.trocr.processing_trocr import TrOCRProcessor

    from samisk_ocr.trocr.types import InputData, ProcessedData


class DatasetSampler(Iterator):
    def __init__(
        self,
        dataset: Sequence[ProcessedData],
        processed_dataset: Sequence[ProcessedData],
        batch_size: int = 8,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.dataset = dataset
        self.processed_dataset = processed_dataset
        self.batch_size = batch_size
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(42)
        self.indices = self.rng.permutation(len(self.dataset))
        self.offset = 0

    def shuffle(self) -> None:
        self.indices = self.rng.permutation(len(self.dataset))

    def __next__(self) -> tuple[InputData, ProcessedData]:
        if self.offset >= len(self.dataset):
            self.offset = 0
            self.shuffle()
        batch = self.dataset[self.indices[self.offset : self.offset + self.batch_size]]
        transformed_batch = self.processed_dataset[
            self.indices[self.offset : self.offset + self.batch_size]
        ]
        self.offset += self.batch_size
        return batch, transformed_batch

    def __iter__(self) -> Iterator[tuple[InputData, ProcessedData]]:
        return self


def transform_data(
    batch: InputData,
    processor: TrOCRProcessor,
    max_target_length: int,
) -> ProcessedData:
    images = [image.convert("RGB") for image in batch["image"]]
    processed_images = processor(images=images, return_tensors="pt").pixel_values
    labels = processor.tokenizer(
        batch["text"], padding="max_length", max_length=max_target_length
    ).input_ids

    # The torch.nn.modules.loss.CrossEntropyLoss has -100 as the default IgnoreIndex
    # So setting the PAD tokens to -100 will make sure they are ignored when we compute the loss
    labels = [
        label if label != processor.tokenizer.pad_token_id else -100 for label in labels
    ]
    return {"pixel_values": processed_images, "labels": labels}
