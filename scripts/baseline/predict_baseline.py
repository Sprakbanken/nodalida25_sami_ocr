import argparse
from functools import lru_cache
from pathlib import Path

import lxml.etree
import pandas as pd
from ipywidgets import interact

from samisk_ocr.clean_text_data import clean
from samisk_ocr.metrics import compute_cer, compute_wer

LANGUAGES = {
    "sma": "Sørsamisk",
    "sme": "Nordsamisk",
    "smj": "Lulesamisk",
    "smn": "Inaresamisk",
}


def urn_col_to_alto_dirname(urncol: str) -> str:
    out = urncol.removeprefix("URN_NBN_no-nb_").removeprefix("no-nb_digavis_").removesuffix("-1")
    if "monografi" in out:
        return out + "_ocr"
    if "digibok" in out:
        return out + "_ocr_xml"
    return out.removeprefix("no-nb_digavis_") + "_ocr_xml"


def urn_col_to_alto_dir(urncol: str) -> Path:
    return Path("data/alto") / urn_col_to_alto_dirname(urncol)


def add_page_to_urn(urn: str, page: int) -> str:
    alto_dirname = urn.removesuffix("_xml").removesuffix("_ocr").removeprefix("URN_NBN_no-nb_")
    if "pliktmonografi" in alto_dirname or "digibok" in alto_dirname:
        return f"{alto_dirname}_{page:04.0f}.xml"
    alto_dirname = alto_dirname.removeprefix("no-nb_digavis_")
    return f"{alto_dirname}_{page:03.0f}_null.xml"


def get_alto_file(df: pd.DataFrame) -> str:
    return df["alto_dir"] / df.apply(
        lambda row: add_page_to_urn(row["urn"], row["page"]), axis="columns"
    ).rename("alto_file")


@lru_cache(1000)
def get_alto_textlines(alto_file: Path) -> list[str]:
    # Parse the XML file
    tree = lxml.etree.parse(alto_file)

    # Find all TextLine elements
    text_lines = tree.xpath("//TextLine")

    # Initialize an empty list to store the concatenated strings
    concatenated_strings = []

    # Iterate over each TextLine element
    for text_line in text_lines:
        # Find all String elements within the current TextLine
        strings = text_line.xpath(".//String")
        # Extract the CONTENT attribute from each String element and join them with spaces
        concatenated_content = " ".join(string.get("CONTENT") for string in strings)

        # Check if the last element is an HYP tag
        last_element = text_line.xpath(".//String|.//HYP")[-1]
        if last_element.tag == "HYP":
            concatenated_content += last_element.get("CONTENT", "")

        # Append the concatenated content to the list
        concatenated_strings.append(concatenated_content)

    # Print the result
    return concatenated_strings


def get_alto_transcription(alto_file, text):
    if not alto_file.exists():
        return pd.NA
    text_lines = get_alto_textlines(alto_file)
    cleaned_text_lines = [clean(l) for l in text_lines]

    return min(cleaned_text_lines, key=lambda l: compute_cer(text, l))


def compute_alto_cer(row):
    return compute_cer(row["text"], row["alto_text_line"])


def compute_alto_wer(row):
    return compute_wer(row["text"], row["alto_text_line"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir / "line_level"
    test_metadata = pd.read_csv("data/new_testset_with_newspapers/test/_metadata.csv")

    test_metadata = (
        test_metadata.assign(alto_dir=test_metadata["urn"].map(urn_col_to_alto_dir))
        .assign(alto_file=get_alto_file)
        .assign(
            transcription=lambda df: df.apply(
                lambda row: get_alto_transcription(row["alto_file"], row["text"]), axis="columns"
            )
        )
    )

    output_csv = output_dir / f"baseline_test.csv"
    test_metadata.to_csv(output_csv, index=False)
