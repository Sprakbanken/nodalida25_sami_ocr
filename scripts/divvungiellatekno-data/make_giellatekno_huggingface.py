import io
import re
from csv import DictWriter
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile

import httpx
from tqdm import tqdm

from samisk_ocr.clean_text_data import clean

COMMIT_HASH = "69c3473feedb616546055fd91cc4a2a032b289ab" # Commmit with corrected transcriptions
DATA_URL = f"https://github.com/divvungiellatekno/tesstrain/archive/{COMMIT_HASH}.zip"


def download_giellatekno_data():
    print("Downloading giellatekno data")
    data_file = Path(__file__).parent / "giellateknodata.zip"
    if data_file.exists():
        print("Data already downloaded")
        return ZipFile(io.BytesIO(data_file.read_bytes()))

    download_file = io.BytesIO()
    with httpx.stream("GET", DATA_URL, follow_redirects=True, timeout=None) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            content_length = int(content_length)
        with tqdm(total=content_length, unit="KiB") as progress:
            for chunk in response.iter_bytes():
                download_file.write(chunk)
                progress.update(len(chunk) // 1024)

    data_file.write_bytes(download_file.getvalue())
    return ZipFile(io.BytesIO(data_file.read_bytes()))


def extract_giellatekno_data(output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    archive = download_giellatekno_data()
    for file in archive.namelist():
        if file.startswith(
            f"tesstrain-{COMMIT_HASH}/training-data/nor_sme-ground-truth"
        ):
            print(f"Extracting {file}")
            archive.extract(file, path=output_path)


def clean_giellatekno_data(txt: str) -> str:
    txt = txt.strip()
    txt = re.sub(r"\s+", " ", txt)
    return clean(txt)


def make_giellatekno_huggingface():
    giellatekno_path = Path("data/giellatekno")
    raw_path = giellatekno_path / "nor_sme-ground-truth-fixed"
    huggingface_path = giellatekno_path / "huggingface_nor_sme-fixed"

    extract_giellatekno_data(raw_path)
    huggingface_path.mkdir(parents=True, exist_ok=True)

    img_path = huggingface_path / "test"
    img_path.mkdir(parents=True, exist_ok=True)
    lines = []
    for img in raw_path.glob("**/*.png"):
        if not img.with_suffix(".gt.txt").exists():
            continue
        print(img)
        new_path = img_path / img.name
        copy2(img, new_path)
        lines.append(
            {
                "file_name": str(new_path.relative_to(huggingface_path)),
                "text": clean_giellatekno_data(img.with_suffix(".gt.txt").read_text()),
            }
        )

    with open(huggingface_path / "metadata.csv", "w") as f:
        writer = DictWriter(f, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(lines)
    with open(huggingface_path / "test/_metadata.csv", "w") as f:
        writer = DictWriter(f, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(
            {"file_name": Path(line["file_name"]).name, "text": line["text"]} for line in lines
        )


if __name__ == "__main__":
    make_giellatekno_huggingface()
