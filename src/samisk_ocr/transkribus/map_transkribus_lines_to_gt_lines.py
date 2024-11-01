import logging
from functools import partial
from pathlib import Path

import pandas as pd

from samisk_ocr.utils import Bbox, image_stem_to_pageurn_line_bbox

logger = logging.getLogger(__name__)


def calculate_overlap_area(bbox1: Bbox, bbox2: Bbox) -> float:
    """Calculate the overlapping area of two Pascal VOC bounding boxes.

    # Citation:
    # The calculation method for the overlapping area was provided by a model-generated chat on OpenAI, 2024.
    """
    # Unpack the bounding boxes
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Calculate the coordinates of the intersection rectangle
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    # Compute the width and height of the intersection rectangle
    intersection_width = max(0, x_inter_max - x_inter_min)
    intersection_height = max(0, y_inter_max - y_inter_min)

    # Calculate the area of the intersection rectangle
    intersection_area = intersection_width * intersection_height

    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)

    # Calculate the overlap as a percentage of bbox1's area
    if bbox1_area > 0:
        overlap_percent = (intersection_area / bbox1_area) * 100
    else:
        overlap_percent = 0

    return overlap_percent


def line_image_dir_to_urn_line_bbox_df(
    image_dir: Path, image_suffixes=[".tif", ".png", ".jpg", ".jpeg"]
) -> pd.DataFrame:
    images = pd.Series([e for suf in image_suffixes for e in image_dir.glob(f"*{suf}")])
    df = pd.DataFrame({"image": images.apply(lambda x: x.name)})
    image_stems = images.apply(lambda x: x.stem)
    page_urns, lines, bboxes = zip(*image_stems.apply(image_stem_to_pageurn_line_bbox))
    df["page_urn"] = page_urns
    df["line"] = lines
    df["bbox"] = bboxes
    return df


def find_image_with_biggest_bbox_overlap(bbox: Bbox, other_df: pd.DataFrame) -> str:
    overlaps = other_df.bbox.apply(partial(calculate_overlap_area, bbox2=bbox))
    return other_df.image[overlaps.argmax()]


def map_transkribus_image_lines_to_gt_image_lines(
    transkribus_df: pd.DataFrame, gt_image_dir: Path
) -> pd.Series:
    """Find the line images from the ground truth image directory that has the largest overlaps with the transkribus line images"""
    gt_df = line_image_dir_to_urn_line_bbox_df(image_dir=gt_image_dir)

    if len(gt_df) != len(transkribus_df):
        logger.warning(
            "Number of lines in transkribus prediction (%s) is not the same as in ground truth (%s)",
            (len(transkribus_df), len(gt_df)),
        )

    transkribus_image_stems = transkribus_df.image.apply(lambda x: Path(x).stem)
    page_urns, lines, bboxes = zip(*transkribus_image_stems.apply(image_stem_to_pageurn_line_bbox))
    transkribus_df["page_urn"] = page_urns
    transkribus_df["line"] = lines
    transkribus_df["bbox"] = bboxes

    dfs = []
    for page_urn, gt_df_ in gt_df.groupby("page_urn"):
        transkribus_df_ = transkribus_df[transkribus_df.page_urn == page_urn]
        assert len(transkribus_df_) == len(gt_df_)

        transkribus_df_ = transkribus_df_.sort_values("line")
        gt_df_ = gt_df_.sort_values("line")

        gt_df_.index = range(len(gt_df_))
        transkribus_df_.index = range(len(transkribus_df_))
        transkribus_df_ = transkribus_df_.rename(columns={"image": "transkribus_image"})

        # set image path to gt_df images when bbox is the same
        same_bbox = gt_df_.bbox == transkribus_df_.bbox
        transkribus_df_["gt_image"] = [None] * len(transkribus_df_)
        transkribus_df_.loc[same_bbox, "gt_image"] = gt_df_[same_bbox].image

        # set image path to the image path with the biggest overlapping bbox to the transkribus bbox
        different_bbox = gt_df_.bbox != transkribus_df_.bbox
        bbox_differs_gt_lines = gt_df_[different_bbox]
        bbox_differs_gt_lines.index = range(len(bbox_differs_gt_lines))

        # Find closest ground truth bbox to each transkribus bbox that differs from ground truth
        for transk_tup in transkribus_df_[different_bbox].itertuples():
            transkribus_df_.at[transk_tup.Index, "gt_image"] = find_image_with_biggest_bbox_overlap(
                bbox=transk_tup.bbox, other_df=bbox_differs_gt_lines
            )

        assert len(set(transkribus_df_.gt_image)) == len(transkribus_df_)
        dfs.append(transkribus_df_)

    map_df = pd.concat(dfs)
    return transkribus_df.merge(map_df, left_on="image", right_on="transkribus_image").gt_image
