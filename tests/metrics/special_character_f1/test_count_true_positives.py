"""
Thus, given the true string "aab" and the transcribed string "abc" and the special characters "abcd",
we would have two true positives ("a" and "b"), one false positive ("c") and one false negative ("a").
A downside with this metric is that it does not take into account the order of the characters, so "abc"
and "bca" would have the same score even though the former is more similar to the ground truth.
"""

from collections import Counter

from samisk_ocr.metrics import SpecialCharacterF1


def test_simple_example():
    ground_truth = "aab"
    transcription = "abc"
    special_characters = "abcd"
    true_character_count = Counter(ground_truth)
    transcription_character_count = Counter(transcription)
    metric = SpecialCharacterF1(special_characters)
    assert metric.count_true_positives(true_character_count, transcription_character_count) == 2


def test_empty_gt():
    ground_truth = ""
    transcription = "abc"
    special_characters = "abcd"
    true_character_count = Counter(ground_truth)
    transcription_character_count = Counter(transcription)
    metric = SpecialCharacterF1(special_characters)
    assert metric.count_true_positives(true_character_count, transcription_character_count) == 0


def test_empty_transcription():
    ground_truth = "aab"
    transcription = ""
    special_characters = "abcd"
    true_character_count = Counter(ground_truth)
    transcription_character_count = Counter(transcription)
    metric = SpecialCharacterF1(special_characters)
    assert metric.count_true_positives(true_character_count, transcription_character_count) == 0
