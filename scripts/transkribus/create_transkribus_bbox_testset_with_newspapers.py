import logging
from functools import partial
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from samisk_ocr.transkribus.export_to_prediction_file import get_line_transcriptions
from samisk_ocr.transkribus.map_transkribus_lines_to_gt_lines import calculate_overlap_area
from samisk_ocr.utils import (
    Bbox,
    get_urn_to_langcode_map,
    image_stem_to_urn_page_line_bbox,
    setup_logging,
)

logger = logging.getLogger(__name__)


def find_row_index_with_biggest_overlap(bbox: Bbox, df: pd.DataFrame) -> int:
    if bbox not in df.columns:
        df = df.copy()
        df.bbox = list(zip(df.xmin, df.ymin, df.xmax, df.ymax))
    overlaps = df.bbox.apply(partial(calculate_overlap_area, bbox2=bbox))
    return int(overlaps.argmax())


def add_urn_page_line_bboxes_to_df(
    df: pd.DataFrame, image_filename_column: str = "image"
) -> pd.DataFrame:
    df = df.copy()
    image_stems = df[image_filename_column].apply(lambda x: Path(x).stem)
    urns, pages, lines, bboxes = zip(*image_stems.apply(image_stem_to_urn_page_line_bbox))

    df["urn"] = urns
    df["page"] = pages
    df["line"] = lines
    df["bbox"] = bboxes

    xmin, ymin, xmax, ymax = zip(*bboxes)

    df["xmin"] = xmin
    df["ymin"] = ymin
    df["xmax"] = xmax
    df["ymax"] = ymax

    return df


if __name__ == "__main__":
    ## Change these as needed ##
    old_dataset_gt_export = Path("data/transkribus_exports/test_data/2997983/Testsett_Samisk_OCR")
    old_dataset_prediction_export = Path(
        "data/transkribus_exports/predictions/test_set/Testsett_Samisk_OCR_testing/"
    )
    newspaper_gt_export = Path("data/transkribus_exports/test_data/aviser")
    newspaper_prediction_export = Path("data/transkribus_exports/predictions/test_set_aviser")
    new_testset_output_p = Path("data/new_testset_with_newspapers/")
    log_level = "DEBUG"
    ############################

    setup_logging(
        source_script="create_transkribus_bbox_testset_with_newspapers", log_level=log_level
    )
    logger.info(
        "Comparing transkribus gt export %s bboxes to transkribus prediction export %s bboxes",
        old_dataset_gt_export,
        old_dataset_prediction_export,
    )

    old_testset_gt_df = get_line_transcriptions(old_dataset_gt_export, keep_source_imgs=True)
    old_testset_prediction_df = get_line_transcriptions(
        old_dataset_prediction_export, keep_source_imgs=True
    )
    old_testset_prediction_df = add_urn_page_line_bboxes_to_df(old_testset_prediction_df)
    old_testset_gt_df = add_urn_page_line_bboxes_to_df(old_testset_gt_df)

    # Remove pliktmonografi-lines
    old_testset_gt_df = old_testset_gt_df[
        old_testset_gt_df.image.apply(lambda x: "pliktmonografi" not in x)
    ]
    old_testset_gt_df.index = range(len(old_testset_gt_df))

    old_testset_prediction_df = old_testset_prediction_df[
        old_testset_prediction_df.image.apply(lambda x: "pliktmonografi" not in x)
    ]
    old_testset_prediction_df.index = range(len(old_testset_prediction_df))

    assert sorted(old_testset_gt_df.urn) == sorted(old_testset_prediction_df.urn)
    assert len(old_testset_gt_df) == len(old_testset_prediction_df)

    logger.info(
        "Comparing transkribus gt export %s bboxes to transkribus prediction export %s bboxes",
        newspaper_gt_export,
        newspaper_prediction_export,
    )
    newspaper_gt_df = get_line_transcriptions(newspaper_gt_export, keep_source_imgs=True)
    newspaper_prediction_df = get_line_transcriptions(
        newspaper_prediction_export, keep_source_imgs=True
    )
    newspaper_gt_df = add_urn_page_line_bboxes_to_df(newspaper_gt_df, image_filename_column="image")
    newspaper_prediction_df = add_urn_page_line_bboxes_to_df(
        newspaper_prediction_df, image_filename_column="image"
    )

    assert len(newspaper_gt_df) == len(newspaper_prediction_df)
    assert sorted(newspaper_gt_df.urn) == sorted(newspaper_prediction_df.urn)

    prediction_df = pd.concat(
        [old_testset_prediction_df, newspaper_prediction_df], ignore_index=True
    )
    gt_df = pd.concat([old_testset_gt_df, newspaper_gt_df], ignore_index=True)

    logger.debug(prediction_df.index)
    logger.debug(gt_df.index)

    same_bbox_df = prediction_df.merge(
        gt_df,
        on=["urn", "page", "line", "xmin", "ymin", "xmax", "ymax", "bbox", "source_image"],
        suffixes=("_pred", "_gt"),
    )

    logger.info("Total number of lines in new testset %s", len(gt_df))
    logger.info(
        "Number of lines where transkribus predictions have the same bbox %s", len(same_bbox_df)
    )

    # Find rows where bboxes are different
    diff = prediction_df.merge(
        gt_df,
        on=["urn", "page", "line", "xmin", "ymin", "xmax", "ymax", "bbox", "source_image"],
        how="outer",
        indicator=True,
        suffixes=("_pred", "_gt"),
    )

    prediction_diff = diff[diff._merge == "left_only"].copy()
    prediction_diff.index = range(len(prediction_diff))

    testset_diff = diff[diff._merge == "right_only"].copy()
    testset_diff.index = range(len(testset_diff))

    assert len(prediction_diff) == len(testset_diff)

    # Keep prediction bboxes, but get text column from testset to create new testset
    # Find (assumed) corresponding textline by calculating maximum area overlap between bboxes
    prediction_diff["text"] = [pd.NA] * len(prediction_diff)
    for (urn, page), df_ in prediction_diff.groupby(["urn", "page"]):
        testset_urn_page = testset_diff[
            (testset_diff.urn == urn) & (testset_diff.page == page)
        ].copy()
        testset_urn_page.index = range(len(testset_urn_page))
        assert len(testset_urn_page) == len(df_)

        bbox_mapping_index = df_.bbox.apply(
            partial(find_row_index_with_biggest_overlap, df=testset_urn_page)
        )
        assert len(bbox_mapping_index) == len(bbox_mapping_index.unique())
        assert len(testset_urn_page.transcription_gt.dropna()) == len(testset_urn_page)

        prediction_diff.loc[df_.index, "text"] = testset_urn_page.transcription_gt[
            bbox_mapping_index
        ].to_list()
        prediction_diff.loc[df_.index, "line"] = testset_urn_page.line[bbox_mapping_index].to_list()

    assert len(prediction_diff.text.dropna()) == len(prediction_diff)

    # Create new testset
    same_bbox_df = same_bbox_df.rename(
        columns={"transcription_gt": "text", "image_pred": "file_name"}
    )
    prediction_diff = prediction_diff.rename(columns={"image_pred": "file_name"})
    columns_to_keep = [
        "file_name",
        "text",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "urn",
        "page",
        "line",
        "source_image",
        "bbox",
    ]

    new_testset_df = pd.concat(
        [same_bbox_df[columns_to_keep], prediction_diff[columns_to_keep]], ignore_index=True
    )
    logger.debug(
        "New dataset length: %s (sum of the two parts: %s)",
        len(new_testset_df),
        len(prediction_diff) + len(same_bbox_df),
    )

    urns_to_langcodes = get_urn_to_langcode_map()
    new_testset_df["langcodes"] = [urns_to_langcodes[e] for e in new_testset_df.urn]

    assert len(new_testset_df.dropna()) == len(new_testset_df)

    inner_testset_df = new_testset_df.copy()

    img_p = new_testset_output_p / "test"
    img_p.mkdir(parents=True)

    newspaper_images = list(newspaper_gt_export.glob("**/*.jpg"))
    for e in tqdm(new_testset_df.itertuples(), total=len(new_testset_df)):
        source_image = old_dataset_gt_export / str(e.source_image)
        if not source_image.exists():
            source_image = [img for img in newspaper_images if img.name == e.source_image][0]

        assert source_image.exists()

        output_img = (
            img_p
            / f"{source_image.stem}_{e.line:03d}_{e.xmin:04d}_{e.ymin:04d}_{e.xmax:04d}_{e.ymax:04d}.jpg"
        )

        img = Image.open(source_image)
        img = img.crop(e.bbox)
        img.save(output_img)
        new_testset_df.at[e.Index, "file_name"] = "test/" + output_img.name
        inner_testset_df.at[e.Index, "file_name"] = output_img.name

    new_testset_df[
        [
            "file_name",
            "text",
            "langcodes",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "urn",
            "page",
            "line",
        ]
    ].to_csv(new_testset_output_p / "metadata.csv", index=False)

    inner_testset_df[
        [
            "file_name",
            "text",
            "langcodes",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "urn",
            "page",
            "line",
        ]
    ].to_csv(new_testset_output_p / "test" / "_metadata.csv", index=False)

    ds = load_dataset(str(new_testset_output_p), split="test")
    logger.debug(ds)
    logger.info("Successfully created new imagefolder dataset at %s", new_testset_output_p)
