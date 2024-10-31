import argparse
import logging
from datetime import datetime
import shutil
from pathlib import Path

import jinja2
import pandas as pd
from PIL import Image


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)

args = parser.parse_args()

args.output.mkdir(parents=True, exist_ok=False)
for split in set(args.input.glob("*/")):
    if split.name not in {"train", "val", "test"}:
        print("Skipping", split)
        continue
    logger.info("Copying %s", split)
    shutil.copytree(split, args.output / split.name)

metadata_file = args.input / "metadata.csv"
shutil.copy(metadata_file, args.output / "metadata.csv")

metadata = pd.read_csv(metadata_file)
for line_num, line in enumerate(metadata.itertuples()):
    image_path: Path = args.input / line.file_name

    with Image.open(image_path) as f:
        width, height = f.size

        pagexml = (
            jinja2.Environment(loader=jinja2.FileSystemLoader(Path(__file__).parent))
            .get_template("./templates/pagexml_template.xml.j2")
            .render(
                timestamp=datetime.now().isoformat(),
                image_file=image_path.name,
                image_width=width,
                image_height=height,
                baseline_y=height,  # left-handed coordinate system (downwards pointing y-axis)
                text=line.text,
            )
        )

        page_dir = (args.output / line.file_name).parent / "page"
        page_filename = Path(line.file_name).with_suffix(".xml").name
        output_file = page_dir / page_filename

        page_dir.mkdir(parents=True, exist_ok=True)
        output_file.write_text(pagexml)
        logger.info("Wrote %s (%d/%d)", output_file, line_num + 1, len(metadata))
