from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch
    from PIL import Image


class InputData(TypedDict):
    image: Sequence[Image.Image]
    transcription: str


class ProcessedData(TypedDict):
    pixel_values: torch.Tensor
    labels: Sequence[int]
