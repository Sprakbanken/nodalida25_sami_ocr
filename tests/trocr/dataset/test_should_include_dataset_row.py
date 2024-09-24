from typing import Any

import hypothesis.strategies as st
from hypothesis import given

from samisk_ocr.trocr.dataset import should_include_dataset_row


@st.composite
def dataset_rows(
    draw,
    width=st.integers(min_value=0),
    height=st.integers(min_value=0),
    text_len=st.integers(min_value=0),
    gt_pix=st.booleans(),
    page_30=st.booleans(),
):
    return {
        "width": draw(width),
        "height": draw(height),
        "text_len": draw(text_len),
        "gt_pix": draw(gt_pix),
        "page_30": draw(page_30),
    }


@given(row=dataset_rows(), min_len=st.integers())
def test_filter_min_len(row, min_len):
    """Rows with text_len < min_len are filtered away, but others are kept."""
    assert (row["text_len"] >= min_len) == should_include_dataset_row(
        row,
        min_len=min_len,
        filter_width=False,
        include_page_30=True,
        include_gt_pix=True,
        min_len_page_30=0,
    )


@given(row=dataset_rows(), filter_width=st.booleans())
def test_filter_width_height_ratio(row, filter_width):
    """Rows with width <= min_with_height_ratio * height are filtered away, but others are kept."""
    assert (row["width"] >= row["height"] or not filter_width) == should_include_dataset_row(
        row,
        min_len=0,
        filter_width=filter_width,
        include_page_30=True,
        include_gt_pix=True,
        min_len_page_30=0,
    )


@given(row=dataset_rows(), include_page_30=st.booleans())
def test_filter_page_30(row, include_page_30):
    """Rows with page_30=True are filtered away if include_page_30=False, but others are kept."""
    assert (not row["page_30"] or include_page_30) == should_include_dataset_row(
        row,
        min_len=0,
        filter_width=False,
        include_page_30=include_page_30,
        include_gt_pix=True,
        min_len_page_30=0,
    )


@given(row=dataset_rows(), include_gt_pix=st.booleans())
def test_filter_gt_pix(row, include_gt_pix):
    """Rows with gt_pix=True are filtered away if include_gt_pix=False, but others are kept."""
    assert (not row["gt_pix"] or include_gt_pix) == should_include_dataset_row(
        row,
        min_len=0,
        filter_width=False,
        include_page_30=True,
        include_gt_pix=include_gt_pix,
        min_len_page_30=0,
    )


@given(row=dataset_rows(), min_len_page_30=st.integers())
def test_filter_min_len_page_30(row, min_len_page_30):
    """Rows with page_30=True and text_len < min_len_page_30 are filtered away, but others are kept."""
    assert (row["text_len"] >= min_len_page_30 or not row["page_30"]) == should_include_dataset_row(
        row,
        min_len=0,
        filter_width=False,
        include_page_30=True,
        include_gt_pix=True,
        min_len_page_30=min_len_page_30,
    )
