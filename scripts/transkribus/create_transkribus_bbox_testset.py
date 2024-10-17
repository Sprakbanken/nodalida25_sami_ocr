import logging
from functools import partial
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from samisk_ocr.transkribus.export_to_prediction_file import get_line_transcriptions
from samisk_ocr.transkribus.map_transkribus_lines_to_gt_lines import calculate_overlap_area
from samisk_ocr.utils import Bbox, image_stem_to_urn_page_line_bbox, setup_logging

logger = logging.getLogger(__name__)


def find_row_index_with_biggest_overlap(bbox: Bbox, df: pd.DataFrame) -> int:
    if bbox not in df.columns:
        df = df.copy()
        df.bbox = list(zip(df.xmin, df.ymin, df.xmax, df.ymax))
    overlaps = df.bbox.apply(partial(calculate_overlap_area, bbox2=bbox))
    return int(overlaps.argmax())


if __name__ == "__main__":
    ## Change these as needed ##
    dataset_p = Path("data/samisk_ocr_dataset/")
    prediction_export = Path(
        "data/transkribus_exports/predictions/test_set/Testsett_Samisk_OCR_testing/"
    )
    new_testset_output_p = Path("data/testset_transkribus_bbox/")
    ############################

    setup_logging(source_script="create_transkribus_bbox_testset", log_level="INFO")
    logger.info(
        "Comparing transkribus export %s bboxes to dataset %s bboxes", prediction_export, dataset_p
    )

    testset_df = pd.read_csv(dataset_p / "test" / "_metadata.csv")
    prediction_df = get_line_transcriptions(prediction_export, keep_source_imgs=True)

    assert len(testset_df) == len(prediction_df)

    transkribus_image_stems = prediction_df.image.apply(lambda x: Path(x).stem)
    urns, pages, lines, bboxes = zip(
        *transkribus_image_stems.apply(image_stem_to_urn_page_line_bbox)
    )

    prediction_df["urn"] = urns
    prediction_df["page"] = pages
    prediction_df["line"] = lines

    prediction_df["bbox"] = bboxes

    xmin, ymin, xmax, ymax = zip(*bboxes)

    prediction_df["xmin"] = xmin
    prediction_df["ymin"] = ymin
    prediction_df["xmax"] = xmax
    prediction_df["ymax"] = ymax

    assert sorted(testset_df.urn) == sorted(prediction_df.urn)

    same_bbox_df = testset_df.merge(
        prediction_df, on=["urn", "page", "line", "xmin", "ymin", "xmax", "ymax"]
    )

    assert sorted(testset_df.urn) == sorted(prediction_df.urn)

    logger.info("Total number of lines in testset %s", len(testset_df))
    logger.info(
        "Number of lines where transkribus predictions have the same bbox %s", len(same_bbox_df)
    )

    # Find rows where bboxes are different
    diff = prediction_df.merge(
        testset_df,
        on=["urn", "page", "line", "xmin", "ymin", "xmax", "ymax"],
        how="outer",
        indicator=True,
    )

    prediction_diff = diff[diff._merge == "left_only"].copy()
    prediction_diff.index = range(len(prediction_diff))

    testset_diff = diff[diff._merge == "right_only"].copy()
    testset_diff.index = range(len(testset_diff))

    assert len(prediction_diff) == len(testset_diff)

    # Keep prediction bboxes, but get text column from testset to create new testset
    # Find (assumed) corresponding textline by calculating maximum area overlap between bboxes
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
        assert len(testset_urn_page) == len(testset_urn_page.text[bbox_mapping_index].dropna())
        prediction_diff.loc[df_.index, "text"] = testset_urn_page.text[bbox_mapping_index].to_list()
        prediction_diff.loc[df_.index, "langcodes"] = testset_urn_page.langcodes[
            bbox_mapping_index
        ].to_list()

    # Create new testset
    new_testset_df = pd.concat([same_bbox_df, prediction_diff], ignore_index=True)
    assert len(new_testset_df.text.dropna()) == len(new_testset_df)

    logger.debug("New testset df length %s", len(new_testset_df))

    img_p = new_testset_output_p / "test"
    img_p.mkdir(parents=True)
    for e in tqdm(new_testset_df.itertuples(), total=len(new_testset_df)):
        source_image = prediction_export / e.source_image
        assert source_image.exists()

        output_img = (
            img_p
            / f"{source_image.stem}_{e.line:03d}_{e.xmin:04d}_{e.ymin:04d}_{e.xmax:04d}_{e.ymax:04d}.jpg"
        )

        img = Image.open(source_image)
        img = img.crop(e.bbox)
        img.save(output_img)
        new_testset_df.at[e.Index, "file_name"] = "test/" + output_img.name

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

    ds = load_dataset(str(new_testset_output_p), split="test")
    logger.debug(ds)
    logger.info("Successfully created new imagefolder dataset at %s", new_testset_output_p)
