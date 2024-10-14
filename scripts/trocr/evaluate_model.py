from argparse import ArgumentParser
from datetime import timedelta
from functools import partial
import logging
from pathlib import Path
from time import monotonic_ns

import datasets
import mlflow

from samisk_ocr.metrics import compute_cer
import samisk_ocr.trocr
from samisk_ocr.utils import setup_logging
from samisk_ocr.trocr.transcribe import transcribe, load_model

parser = ArgumentParser()
parser.add_argument("--run_name", type=str, required=True, help="The name of the run to evaluate, e.g. 'respected-rat-222'")
parser.add_argument("--experiment_name", type=str, required=True, help="The name of the experiment to evaluate, e.g. 'TrOCR trocr-base-printed finetuning'")
args = parser.parse_args()

config = samisk_ocr.trocr.config.Config()
mlflow.set_tracking_uri(config.mlflow_url)
mlflow.set_experiment(args.experiment_name)

logger = logging.getLogger(__name__)
setup_logging(source_script=Path(__file__).stem, log_level="INFO")

# Load the datasets
logger.info("Loading validation data")
validation_set = datasets.load_dataset("imagefolder", data_dir=config.DATA_PATH, split="validation")

# Specify what model we want to load
run_name = args.run_name
all_checkpoints_dir = Path(f"data/checkpoints/train_trocr_base_printed/{run_name}/")
run_id = mlflow.search_runs(filter_string=f"run_name = '{run_name}'")["run_id"].item() # type: ignore

with mlflow.start_run(run_id=run_id) as run:
    cer_values = {}

    # load processor and model
    all_checkpoints = sorted(all_checkpoints_dir.glob("checkpoint-*"), key=lambda x: int(x.name.partition("-")[2]))
    for i, checkpoint_dir in enumerate(all_checkpoints, start=1):
        logger.info("Loading and evaluating model, %s (%d / %d)", checkpoint_dir, i, len(all_checkpoints))
        t0 = monotonic_ns()
        processor, model = load_model(checkpoint_dir, processor_name=all_checkpoints_dir / "processor")

        validation_set_with_estimates = validation_set.map(partial(transcribe, processor=processor, model=model), batched=True, batch_size=32)

        # Compute CER and check if it is the best so far
        true = "".join(validation_set_with_estimates["text"])
        pred = "".join(validation_set_with_estimates["transcription"])
        cer_values[checkpoint_dir.name] = compute_cer(true, pred)

        mlflow.log_dict(
            {
                "ground_truth": validation_set_with_estimates["text"],
                "transcription": validation_set_with_estimates["transcription"],
                "urn": validation_set_with_estimates["urn"],
                "page": validation_set_with_estimates["page"],
                "line": validation_set_with_estimates["line"],
            },
            str(Path("final_evaluation") / f"checkpoints/{checkpoint_dir.name}.json"),
        )

        t1 = monotonic_ns()
        duration = timedelta(seconds=round((t1 - t0) * 1e-9))
        logger.info("Evaluation time - %s, cer - %02f", duration, cer_values[checkpoint_dir.name] * 100)

    mlflow.log_dict(
        {
            "cer_values": cer_values,
            "min_cer": min(cer_values.values()),
            "min_cer_checkpoint": min(cer_values.keys(), key=cer_values.__getitem__)
        },
        str(Path("final_evaluation") / "min_cer.json"),
    )
