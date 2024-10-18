import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from samisk_ocr.utils import setup_logging

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def split_to_datasets_split(split: str) -> str:
    if split == "val":
        return "validation"
    return split


def transcribe(
    batch,
    processor,
    model,
) -> dict[str, list[str]]:
    rgb_images = [image.convert("RGB") for image in batch["image"]]
    processed_images = processor(images=rgb_images, return_tensors="pt").pixel_values
    tokens = model.generate(processed_images.to(model.device))
    texts = processor.batch_decode(tokens, skip_special_tokens=True)
    return {"transcription": texts}


def load_model(model_name: str, processor_name: str | None = None) -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    if not processor_name:
        processor_name = model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(processor_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    model.generation_config.use_cache = True

    assert model.config.decoder_start_token_id == processor.tokenizer.cls_token_id
    assert model.config.pad_token_id == processor.tokenizer.pad_token_id
    assert model.config.vocab_size == model.config.decoder.vocab_size
    return processor, model


def transcribe_dataset(
    model_name: str, dataset_path: Path, split: str, batch_size: int
) -> pd.DataFrame:
    """Run TrOCR model on all images in dataset split"""
    df = pd.read_csv(dataset_path / split / "_metadata.csv")

    processor, model = load_model(model_name)
    logger.info("TrOCR model generation config %s", model.generation_config)

    ds = load_dataset(str(dataset_path), split=split_to_datasets_split(split))
    ds_with_transcriptions = ds.map(
        partial(transcribe, processor=processor, model=model),
        batched=True,
        batch_size=batch_size,
    )

    ds_df = ds_with_transcriptions.to_pandas()
    ds_df = ds_df[["transcription", "urn", "page", "line"]]

    df_with_transcriptions = df.merge(ds_df, on=["urn", "page", "line"], validate="1:1")
    assert len(df_with_transcriptions) == len(df)

    return df_with_transcriptions


if __name__ == "__main__":
    parser = ArgumentParser(description="Transcribe with TrOCR model")
    parser.add_argument(
        "model_name",
        help="Path to local model or model id on huggingface hub",
    )
    parser.add_argument(
        "--dataset",
        help="Path to local dataset",
        type=Path,
        default=Path("data/samisk_ocr_dataset/"),
    )
    parser.add_argument("--split", default="val", help="Dataset split to transcribe")
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="The output directory to store predicted transcriptions",
        default=Path("output/predictions/"),
    )
    parser.add_argument("--batch_size", type=int, help="Batch size when transcribing", default=16)
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    args = parser.parse_args()
    setup_logging(source_script="trocr_transcribe", log_level=args.log_level)

    df = transcribe_dataset(
        model_name=args.model_name,
        dataset_path=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
    )

    output_dir = args.output_dir / "line_level"

    output_dir.mkdir(exist_ok=True, parents=True)
    output_csv = output_dir / f"{Path(args.model_name).name}_{args.split}.csv"
    df.to_csv(output_csv, index=False)

    logger.info(f"Wrote predicted transcriptions to {output_csv}")
