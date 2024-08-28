from collections.abc import Sequence
from typing import Protocol

import datasets.arrow_dataset


class Evaluator(Protocol):
    def __call__(
        self,
        data: datasets.arrow_dataset.Dataset,
        pred_texts: list[str],
        step: int,
        key_prefix: str,
    ) -> None: ...


class Metric(Protocol):
    def __call__(self, prediction: str, reference: str) -> float: ...


class ReductionFunction(Protocol):
    def __call__(self, values: Sequence[float]) -> float: ...
