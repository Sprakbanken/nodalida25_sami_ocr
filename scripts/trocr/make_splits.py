import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

rng = np.random.default_rng(42)

dataset = load_dataset("imagefolder", data_dir="/media/data/hf_dataset", split="train")


# Hent ut delen av datasettet som er kurert
curated_dataset = dataset.filter(lambda x: not x["page_30"])

# Hent ut URN-ene
curated_urns = set(curated_dataset["urn"])
uncurated_urns = set(dataset["urn"]) - curated_urns

# Sett opp andel trenings- og valideringsdata
val_fraction = 0.25

num_curated_val = round(val_fraction * len(curated_urns))
num_uncurated_val = round(val_fraction * len(uncurated_urns))

# Sample URN-ene fra det kurerte og ukurerte datasettet
curated_validation_urns = rng.choice(list(curated_urns), num_curated_val, replace=False)
uncurated_validation_urns = rng.choice(list(uncurated_urns), num_uncurated_val, replace=False)

curated_training_urns = list(curated_urns - set(curated_validation_urns))
uncurated_training_urns = list(uncurated_urns - set(uncurated_validation_urns))

out = {
    "train": [*uncurated_training_urns, *curated_training_urns],
    "val": [*uncurated_validation_urns, *curated_validation_urns],
}

# Lagre URN-ene til fil
output_file = Path(__file__).parent.parent / "data/urns.json"
output_file.parent.mkdir(exist_ok=True, parents=True)
output_file.write_text(json.dumps(out))

# Filtrer datasettet for å få trening- og valideringsdata
train_set = dataset.filter(lambda x: x["urn"] in set(out["train"]))
val_set = dataset.filter(lambda x: x["urn"] in set(out["val"]))

curated_train_set = train_set.filter(lambda x: not x["page_30"])
curated_val_set = val_set.filter(lambda x: not x["page_30"])

print("       METADATA", "TRAIN", "VAL", "TOTAL")
print("   FULL URNCOUNT", len(out["train"]), len(out["val"]), len(out["train"]) + len(out["val"]))
print(
    "CURATED URNCOUNT",
    len(curated_training_urns),
    len(curated_validation_urns),
    len(curated_training_urns) + len(curated_validation_urns),
)
print("   FULL DATASIZE", len(train_set), len(val_set), len(train_set) + len(val_set))
print(
    "CURATED DATASIZE",
    len(curated_train_set),
    len(curated_val_set),
    len(curated_train_set) + len(curated_val_set),
)
