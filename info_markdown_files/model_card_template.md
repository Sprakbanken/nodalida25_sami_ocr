---
library_name: transformers
license: cc-by-4.0
language:
  - smi
  - smj
  - sme
  - sma
  - smn
datasets:
  - Sprakbanken/synthetic_sami_ocr_data
base_model:
  - microsoft/trocr-base-printed
---

# Model Card for Model ID
This is a TrOCR-model for OCR (optical character recognition) of Sámi languages.
It can be used to recognize text in images of printed text (scanned books, magazines, etc.) in North Sámi, South Sámi, Lule Sámi, and Inari Sámi.

## Collection details 
This model is a part of our collection of OCR models for Sámi languages.

The following TrOCR models are available:
- [Sprakbanken/trocr_smi](https://huggingface.co/Sprakbanken/trocr_smi): [microsoft/trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed) fine-tuned on manually annotated Sámi data
- [Sprakbanken/trocr_smi_nor](https://huggingface.co/Sprakbanken/trocr_smi_nor): microsoft/trocr-base-printed fine-tuned on manually annotated Sámi and Norwegian data
- [Sprakbanken/trocr_smi_pred](https://huggingface.co/Sprakbanken/trocr_smi_pred): microsoft/trocr-base-printed fine-tuned on manually annotated and automatically transcribed Sámi data
- [Sprakbanken/trocr_smi_nor_pred](https://huggingface.co/Sprakbanken/trocr_smi_nor_pred): microsoft/trocr-base-printed fine-tuned on manually annotated and automatically transcribed Sámi data, and manually annotated Norwegian data
- [Sprakbanken/trocr_smi_synth](https://huggingface.co/Sprakbanken/trocr_smi_synth): microsoft/trocr-base-printed fine-tuned on [Sprakbanken/synthetic_sami_ocr_data](https://huggingface.co/datasets/Sprakbanken/synthetic_sami_ocr_data), and then on manually annotated Sámi data
- [Sprakbanken/trocr_smi_pred_synth](https://huggingface.co/Sprakbanken/trocr_smi_pred_synth): microsoft/trocr-base-printed fine-tuned on Sprakbanken/synthetic_sami_ocr_data, and then fine-tuned on manually annotated and automatically transcribed Sámi data
- [Sprakbanken/trocr_smi_nor_pred_synth](https://huggingface.co/Sprakbanken/trocr_smi_nor_pred_synth): microsoft/trocr-base-printed fine-tuned on Sprakbanken/synthetic_sami_ocr_data, and then fine-tuned on manually annotated and automatically transcribed Sámi data, and manually annotated Norwegian

[Sprakbanken/trocr_smi_pred_synth](https://huggingface.co/Sprakbanken/trocr_smi_pred_synth) is the model that achieved the best results (of the TrOCR models) on our test dataset.

## Model Details
<!-- model details here  -->

### Model Description

- **Developed by:** The National Library of Norway
- **Model type:** TrOCR
- **Languages:**  North Sámi (sme), South Sámi (sma), Lule Sámi (smj), and Inari Sámi (smn)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Finetuned from model :** [microsoft/trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed)

### Model Sources 

- **Repository:** https://github.com/Sprakbanken/nodalida25_sami_ocr
- **Paper:** "Enstad T, Trosterud T, Røsok MI, Beyer Y, Roald M. Comparative analysis of optical character recognition methods for Sámi texts from the National Library of Norway. Accepted for publication in Proceedings of the 25th Nordic Conference on Computational Linguistics (NoDaLiDa) 2025." (preprint coming soon.)

## Uses
You can use the raw model for optical character recognition (OCR) on single text-line images in North Sámi, South Sámi, Lule Sámi, and Inari Sámi. 

### Out-of-Scope Use
The model only works with images of lines of text.
If you have images of entire pages of text, you must segment the text into lines first to benefit from this model.


## How to Get Started with the Model

Use the code below to get started with the model.

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("Sprakbanken/trocr_smi_pred_synth")
model = VisionEncoderDecoderModel.from_pretrained("Sprakbanken/trocr_smi_pred_synth")

image = Image.open("path_to_image.jpg").convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```
## Citation 

**APA:**

Enstad, T., Trosterud, T., Røsok, M. I., Beyer, Y., & Roald, M. (2025). Comparative analysis of optical character recognition methods for Sámi texts from the National Library of Norway. Proceedings of the 25th Nordic Conference on Computational Linguistics (NoDaLiDa).
