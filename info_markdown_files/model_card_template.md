---
library_name: transformers
tags: []
---

# Model Card for Model ID
This is a TrOCR-model for OCR (optical character recognition) of Sámi languages.
It can be used to recognize text in images of printed text (scanned books, magazines, etc.) in North Sámi, South Sámi, Lule Sámi, and Inari Sámi.

## Model Details

### Model Description

- **Developed by:** The National Library of Norway
- **Model type:** TrOCR
- **Languages:**  North Sámi (sme), South Sámi (sma), Lule Sámi (smj), and Inari Sámi (smn)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Finetuned from model :** [TrOCR-printed base model](https://huggingface.co/microsoft/trocr-base-printed)

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
