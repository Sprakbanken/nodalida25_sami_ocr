# Comparative analysis of optical character recognition methods for Sámi texts from the National Library of Norway
This repository contains the supplementary materials for "Enstad T, Trosterud T, Røsok MI, Beyer Y, Roald M. Comparative analysis of optical character recognition methods for Sámi texts from the National Library of Norway. Accepted for publication in Proceedings of the 25th Nordic Conference on Computational Linguistics (NoDaLiDa) 2025." (preprint coming soon.)

## Models

### TrOCR models
The models can be found on our [huggingface page](https://huggingface.co/Sprakbanken)

- [trocr_smi](https://huggingface.co/Sprakbanken/trocr_smi) is the [TrOCR-printed base model](https://huggingface.co/microsoft/trocr-base-printed) fine-tuned on GT-Sámi
- [trocr_smi_nor](https://huggingface.co/Sprakbanken/trocr_smi_nor) is the is the TrOCR-printed base model fine-tuned on GT-Sámi and GT-Nor
- [trocr_smi_pred](https://huggingface.co/Sprakbanken/trocr_smi_pred) is the TrOCR-printed base model fine-tuned on GT-Sámi and Pred-Sámi
- [trocr_smi_nor_pred](https://huggingface.co/Sprakbanken/trocr_smi_nor_pred) is the TrOCR-printed base model fine-tuned on GT-Sámi, GT-Nor and Pred-Sámi
- [trocr_smi_synth](https://huggingface.co/Sprakbanken/trocr_smi_synth) is the TrOCR-printed base model fine-tuned on Synth-Sámi (5 epochs), and then fine-tuned on GT-Sámi
- [trocr_smi_pred_synth](https://huggingface.co/Sprakbanken/trocr_smi_pred_synth) is the TrOCR-printed base model fine-tuned on Synth-Sámi (5 epochs), and then fine-tuned on GT-Sámi and Pred-Sámi
- [trocr_smi_nor_pred_synth](https://huggingface.co/Sprakbanken/trocr_smi_nor_pred_synth) is the TrOCR-printed base model fine-tuned on Synth-Sámi (5 epochs), and then fine-tuned on GT-Sámi, GT-Nor and Pred-Sámi


### Tesseract models
- [ub_smi](tesseract_models/ub_smi) is tesseract model trained from scratch on GT-Sámi (the first row in table 3)
- [smi](tesseract_models/smi) is [the Norwegian tesseract base model](https://github.com/tesseract-ocr/tessdata_best/blob/main/nor.traineddata) fine-tuned on GT-Sámi
- [smi_nor](tesseract_models/smi_nor) is the Norwegian tesseract base model fine-tuned on GT-Sámi and GT-Nor
- [smi_pred](tesseract_models/smi_pred) is the Norwegian tesseract base model fine-tuned on GT-Sámi and Pred-Sámi
- [smi_nor_pred](tesseract_models/smi_nor_pred) is the Norwegian tesseract base model fine-tuned on GT-Sámi, GT-Nor and Pred-Sámi
- [synth_base](tesseract_models/synth_base) is the Norwegian tesseract base model fine-tuned on Synth-Sámi
- [sb_smi](tesseract_models/sb_smi) is synth_base fine-tuned on GT-Sámi
- [sb_smi_nor_pred](tesseract_models/sb_smi_nor_pred) is synth_base fine-tuned GT-Sámi, GT-Nor and Pred-Sámi

See [tesseract_models/README.md](tesseract_models/README.md) for more details

### Transkribus models
The transkribus models can be found in the transkribus app.

Here is the list of transkribus model names and their details
- SamiskOCR_smi_ub (model id 181605): the transkribus model trained on GT-Sámi without a base model (first row in table 3)
- SamiskOCR_smi (model id 181725): the Transkribus print M1 base model (model id 39995) fine-tuned on GT-Sámi
- SamiskOCR_smi_nor (model id 182005): the Transkribus print M1 base model fine-tuned on GT-Sámi and GT-Nor
- SamiskOCR_smi_smipred (model id 192137): the Transkribus print M1 base model fine-tuned on GT-Sámi and Pred-Sámi
- SamiskOCR_alt (model id 179305): the Transkribus print M1 base model fine-tuned on GT-Sámi, GT-Nor and Pred-Sámi


## Installation
You can make a virtual Python environment and install like this:
(if you have the correct version of Python and venv installed)
```
python3 -m venv env-name
. env-name/bin/activate
pip install .
```
Then you can run all scripts in src/samisk_ocr like this
```
python3 -m samisk_ocr.evaluate_predictions <arg1> <arg2> ..
python3 -m samisk_ocr.tesseract.transcribe <arg1> <arg2> ..
```

Or use tools like pdm:
```
pdm install
pdm run python -m samisk_ocr.evaluate_predictions <arg1> <arg2> ..
pdm run python -m samisk_ocr.tesseract.transcribe <arg1> <arg2> ..
```

The code to make the tables and plots can be found in [notebooks/tables](notebooks/tables) and [notebooks/plots](notebooks/plots)
The code to train models can be found in [src/samisk_ocr/trocr](src/samisk_ocr/trocr) and [src/samisk_ocr/tesseract](src/samisk_ocr/tesseract)
The code to evaluate model predictions can be found in [src/samisk_ocr/evaluate_predictions.py](src/samisk_ocr/evaluate_predictions.py)
