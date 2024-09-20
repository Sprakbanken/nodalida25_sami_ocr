import logging
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

from samisk_ocr.clean_text_data import clean
from samisk_ocr.utils import setup_logging

logger = logging.getLogger(__name__)


def clean_xml_tree(xml_tree: ET.ElementTree) -> ET.ElementTree:
    tree = deepcopy(xml_tree)
    root = tree.getroot()
    namespace = ""
    if "}" in root.tag:
        namespace = root.tag[: root.tag.index("}") + 1]
    string_tag = namespace + "String"

    for elem in root.iter():
        if elem.tag == string_tag:
            content = elem.get("CONTENT", "")
            elem.set("CONTENT", clean(content))

    return tree


def clean_xml_file(input_file: Path, output_file: Path):
    xml_tree = ET.parse(input_file)
    clean_tree = clean_xml_tree(xml_tree=xml_tree)
    clean_tree.write(output_file, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Replace unwanted characters in text content in alto xml files in directory"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="The directory containing xmlfiles",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="If set, will copy clean files to output dir (else cleans input directory in-place)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    args = parser.parse_args()
    setup_logging(source_script="clean_alto_xml", log_level=args.log_level)

    output_dir = args.output_dir if args.output_dir else args.input_dir

    for xml_file in args.input_dir.glob("**/alto/*.xml"):
        output_parent = output_dir / xml_file.parent.relative_to(args.input_dir)
        output_parent.mkdir(parents=True, exist_ok=True)

        output_file = output_parent / xml_file.name
        clean_xml_file(input_file=xml_file, output_file=output_file)

    logger.info("See all clean alto xml_files in %s", output_dir)
