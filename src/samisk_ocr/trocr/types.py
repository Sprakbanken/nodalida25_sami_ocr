from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import torch
    from PIL import Image


class InputData(TypedDict):
    image: list[Image.Image]
    transcription: str


class TransformedData(TypedDict):
    pixel_values: torch.Tensor
    labels: list[int]
