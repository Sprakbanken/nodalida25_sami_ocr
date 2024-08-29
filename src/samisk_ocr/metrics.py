from collections import Counter
from typing import Protocol

from jiwer import cer, wer


class Metric(Protocol):
    def __call__(self, ground_truth: str, transcription: str, **_) -> float: ...

    # We have **_ to allow for additional keyword arguments to be passed to the function
    # this makes it possible to apply a metric on any data frame as long as it has the
    # columns "ground_truth" and "transcription":
    # df.apply(lambda row: metric(**row), axis=1)


def compute_cer(ground_truth: str, transcription: str) -> float:
    return cer(hypothesis=transcription, reference=ground_truth)


def compute_wer(ground_truth: str, transcription: str) -> float:
    return wer(hypothesis=transcription, reference=ground_truth)


class SpecialCharacterF1:
    """Compute the F1 score for special characters only.

    To count the true positives, we count the number of characters that occur equally often in
    the ground truth and the transcription and weight it by the number of times the character occurs.
    The number of false positives is the number of extra occurences of characters in the transcription
    compared to the ground truth and the number of false negatives is the number of extra occurences of
    characters in the ground truth compared to the transcription.

    Thus, given the true string "aab" and the transcribed string "abc" and the special characters "abcd",
    we would have two true positives ("a" and "b"), one false positive ("c") and one false negative ("a").
    A downside with this metric is that it does not take into account the order of the characters, so "abc"
    and "bca" would have the same score even though the former is more similar to the ground truth.

    This also means that the same character cannot have a nonzero true positive and a nonzero false positive.
    """

    def __init__(self, special_characters: str):
        self.special_characters = set(special_characters)

    def __call__(self, ground_truth: str, transcription: str) -> float:
        true_character_count = Counter(ground_truth)
        transcription_character_count = Counter(transcription)

        true_positives = self.count_true_positives(
            true_character_count, transcription_character_count
        )
        false_positives = self.count_false_positives(
            true_character_count, transcription_character_count
        )
        false_negatives = self.count_false_negatives(
            true_character_count, transcription_character_count
        )

        return 2 * true_positives / (2 * true_positives + (false_positives + false_negatives))

    def count_true_positives(
        self, ground_truth_counter: Counter, transcription_counter: Counter
    ) -> int:
        return sum(
            min(ground_truth_counter[char], transcription_counter[char])
            for char in self.special_characters
        )

    def count_false_positives(
        self, ground_truth_counter: Counter, transcription_counter: Counter
    ) -> int:
        return sum(
            max(0, transcription_counter[char] - ground_truth_counter[char])
            for char in self.special_characters
        )

    def count_false_negatives(
        self, ground_truth_counter: Counter, transcription_counter: Counter
    ) -> int:
        return sum(
            max(0, ground_truth_counter[char] - transcription_counter[char])
            for char in self.special_characters
        )
