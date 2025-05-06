# Comparative analysis of optical character recognition methods for Sámi texts from the National Library of Norway
This repository contains the supplementary materials for "Enstad T, Trosterud T, Røsok MI, Beyer Y, Roald M. Comparative analysis of optical character recognition methods for Sámi texts from the National Library of Norway. Proceedings of the Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies (NoDaLiDa/Baltic-HLT 2025)"

Read the paper here: [https://aclanthology.org/2025.nodalida-1.11/](https://aclanthology.org/2025.nodalida-1.11/)

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
You can try out our best model [here](https://www.transkribus.org/model/sami-ocr)

Here is the list of transkribus model names and their details
- SamiskOCR_smi_ub (model id 181605): the transkribus model trained on GT-Sámi without a base model (first row in table 3)
- SamiskOCR_smi (model id 181725): the Transkribus print M1 base model (model id 39995) fine-tuned on GT-Sámi
- SamiskOCR_smi_nor (model id 182005): the Transkribus print M1 base model fine-tuned on GT-Sámi and GT-Nor
- SamiskOCR_smi_smipred (model id 192137): the Transkribus print M1 base model fine-tuned on GT-Sámi and Pred-Sámi
- SamiskOCR_alt (model id 179305): the Transkribus print M1 base model fine-tuned on GT-Sámi, GT-Nor and Pred-Sámi

## Synthetic data
The code to generate the synthetic data (Synth-Sámi) we used to pre-train Tesseract and TrOCR models is available in this [GitHub repository](https://github.com/sprakbanken/synthetic_text_images), and the dataset is [available on Huggingface](https://huggingface.co/datasets/Sprakbanken/synthetic_sami_ocr_data).

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


## Citation


```
@inproceedings{enstad-etal-2025-comparative,
    title = "Comparative analysis of optical character recognition methods for {S{\'a}mi} texts from the National Library of {Norway}",
    author = "Enstad, Tita  and
      Trosterud, Trond  and
      R{\o}sok, Marie Iversdatter  and
      Beyer, Yngvil  and
      Roald, Marie",
    editor = "Johansson, Richard  and
      Stymne, Sara",
    booktitle = "Proceedings of the Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies (NoDaLiDa/Baltic-HLT 2025)",
    month = mar,
    year = "2025",
    address = "Tallinn, Estonia",
    publisher = "University of Tartu Library",
    url = "https://aclanthology.org/2025.nodalida-1.11/",
    pages = "98--108",
    ISBN = "978-9908-53-109-0",
    abstract = "Optical Character Recognition (OCR) is crucial to the National Library of Norway`s (NLN) digitisation process as it converts scanned documents into machine-readable text. However, for the S{\'a}mi documents in NLN`s collection, the OCR accuracy is insufficient. Given that OCR quality affects downstream processes, evaluating and improving OCR for text written in S{\'a}mi languages is necessary to make these resources accessible. To address this need, this work fine-tunes and evaluates three established OCR approaches, Transkribus, Tesseract and TrOCR, for transcribing S{\'a}mi texts from NLN`s collection. Our results show that Transkribus and TrOCR outperform Tesseract on this task, while Tesseract achieves superior performance on an out-of-domain dataset. Furthermore, we show that fine-tuning pre-trained models and supplementing manual annotations with machine annotations and synthetic text images can yield accurate OCR for S{\'a}mi languages, even with a moderate amount of manually annotated data."
}
``` 