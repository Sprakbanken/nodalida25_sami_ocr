# OCR-modeller for samiske språk
Dette er en samling av modeller for OCR (optical character recognition) av samiske språk.
Disse kan brukes til å gjenkjenne tekst i bilder av trykt tekst (scannede bøker, magasiner, o.l) på nordsamisk, sørsamisk, lulesamisk og inaresamisk.

Mer detaljert informasjon om trening og evaluering av modellene kan du lese i artikkelen [Comparative analysis of optical character recognition methods for Sámi texts from the National Library of Norway](url.no)

Samlingen består tre forskjellige typer modeller: Transkribus-modeller, Tesseract-modeller og TrOCR-modeller.

## Transkribus-modeller
Transkribusmodellene er tilgjengelige i applikasjonen [Transkribus](https://www.transkribus.org/).
Transkribus er et verktøy hvor du kan bruke modeller til å gjenkjenne tekst i bilder, og korrigere eventuelle feilprediksjoner.
Det kan koste penger å bruke Transkribus, avhengig av mengden data som skal transkriberes.

Følgende modeller er tilgjengelige:
- SamiskOCR_smi_ub (modell-id 181605): en modell trent utelukkende på manuelt annotert samisk data
- SamiskOCR_smi (modell-id 181725): print M1-basemodellen (modell-id 39995) fin-tunet på manuelt annotert samisk data
- SamiskOCR_smi_nor (modell-id 182005): print M1-basemodellen fin-tunet på manuelt annotert samisk og norsk data
- SamiskOCR_smi_smipred (modell-id 192137): print M1-basemodellen fin-tunet på manuelt annotert og automatisk transkribert samisk data
- SamiskOCR_alt (modell-id 179305): print M1-basemodellen fin-tunet på manuelt annotert og automatisk transkribert samisk data, og manuelt annotert norsk data

Sistnevnte modell er den som fikk best resultater på vårt testdatasett.

## Tesseract-modeller
Tesseract er et OCR-verktøy hvor du kan kjøre modeller lokalt.
Se [installasjonsguiden](https://tesseract-ocr.github.io/tessdoc/Installation.html) for å installere Tesseract.
Så snart du har installert Tesseract, kan du bruke .traineddata-modellfilene som vi deler her, for å kjøre OCR på bildene dine.


Følgende modeller er tilgjengelige:
- [ub_smi](tesseract_models/ub_smi): en modell trent utelukkende på manuelt annotert samisk data
- [smi](tesseract_models/smi): [den norske basemodellen](https://github.com/tesseract-ocr/tessdata_best/blob/main/nor.traineddata) fin-tunet på manuelt annotert samisk data
- [smi_nor](tesseract_models/smi_nor): den norske basemodellen fin-tunet på manuelt annotert samisk og norsk data
- [smi_pred](tesseract_models/smi_pred): den norske basemodellen fin-tunet på manuelt annotert og automatisk transkribert samisk data
- [smi_nor_pred](tesseract_models/smi_nor_pred): den norske basemodellen fin-tunet på manuelt annotert og automatisk transkribert samisk data, og manuelt annotert norsk data
- [synth_base](tesseract_models/synth_base): den norske basemodellen fin-tunet på syntetisk* samisk data
- [sb_smi](tesseract_models/sb_smi): synth_base fin-tunet på manuelt annotert samisk data
- [sb_smi_nor_pred](tesseract_models/sb_smi_nor_pred): synth_base fin-tunet på manuelt annotert og automatisk transkribert samisk data, og manuelt annotert norsk data

Sistnevnte modell er den som fikk best resultater (av tesseract-modellene) på vårt testdatasett

\* syntetisk her betyr at vi har ekte samisk tekst, som vi kan laget bilder av, som skal ligne på scannet tekst. I motsetning til den manuelt transkriberte dataen, som er bøker og aviser som er scannet, og deretter manuelt transkribert.

Tesseract-tips:
- Du må flytte .traineddata-filene til tessdata-området på PC-en din
- For å kjøre OCR med ønsket modell må du spesifisere modellnavnet med -l, eks: `tesseract filnavn.jpg utfil -l smi_nor_pred`. Da vil `utfil.txt` inneholde teksten som modellen fant i bildet
- Bruk --psm 7 hvis du har bilde av en linje med tekst, og ikke en hel side. Se mer info [her](https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc)

## TrOCR-modeller
TrOCR er en transformers-basert modellarkitektur for tekstgjenkjenning.
Du kan bruke modellene med [transformers-biblioteket](https://huggingface.co/docs/transformers/v4.47.1/en/model_doc/trocr#inference)

Følgende modeller er tilgjengelige:
- [trocr_smi](https://huggingface.co/Sprakbanken/trocr_smi): [TrOCR-printed-basemodellen](https://huggingface.co/microsoft/trocr-base-printed) fin-tunet på manuelt annotert samisk data
- [trocr_smi_nor](https://huggingface.co/Sprakbanken/trocr_smi_nor): TrOCR-printed-basemodellen fin-tunet on på manuelt annotert samisk og norsk data
- [trocr_smi_pred](https://huggingface.co/Sprakbanken/trocr_smi_pred): TrOCR-printed-basemodellen fin-tunet på manuelt annotert og automatisk transkribert samisk data
- [trocr_smi_nor_pred](https://huggingface.co/Sprakbanken/trocr_smi_nor_pred): TrOCR-printed-basemodellen fin-tunet på manuelt annotert og automatisk transkribert samisk data, og manuelt annotert norsk data
- [trocr_smi_synth](https://huggingface.co/Sprakbanken/trocr_smi_synth): TrOCR-printed-basemodellen fin-tunet på syntetisk* samisk data, og deretter på manuelt annotert samisk data
- [trocr_smi_pred_synth](https://huggingface.co/Sprakbanken/trocr_smi_pred_synth): trocr_smi_synth fin-tunet på manuelt annotert og automatisk transkribert samisk data
- [trocr_smi_nor_pred_synth](https://huggingface.co/Sprakbanken/trocr_smi_nor_pred_synth): trocr_smi_synth fin-tunet på manuelt annotert og automatisk transkribert samisk data, og manuelt annotert norsk

trocr_smi_pred_synth er modellen som fikk best resultater (av TrOCR-modellene) på vårt testdatasett

\* syntetisk her betyr at vi har ekte samisk tekst, som vi kan laget bilder av, som skal ligne på scannet tekst. I motsetning til den manuelt transkriberte dataen, som er bøker og aviser som er scannet, og deretter manuelt transkribert.


Modellene fungerer kun med bilder av linjer av tekst.
Om du har bilder av hele sider av tekst, må du dermed segmentere teksten i linjer først, for å få nytte av denne modellen.

Brukseksempel med python
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

# OCR Models for Sámi Languages
This is a collection of models for OCR (optical character recognition) of Sámi languages.
These can be used to recognize text in images of printed text (scanned books, magazines, etc.) in Northern Sámi, Southern Sámi, Lule Sámi, and Inari Sámi.

You can read more detailed information about the training and evaluation of the models in the article [Comparative analysis of optical character recognition methods
for Sámi texts from the National Library of Norway](url.no)

The collection consists of three different types of models: Transkribus models, Tesseract models, and TrOCR models.

## Transkribus Models
The Transkribus models are available in the application [Transkribus](https://www.transkribus.org/).
Transkribus is a tool where you can use models to recognize text in images and correct any prediction errors.
It may cost money to use Transkribus, depending on the amount of data to be transcribed.

The following models are available:
- SamiskOCR_smi_ub (model-id 181605): a model trained exclusively on manually annotated Sámi data
- SamiskOCR_smi (model-id 181725): print M1 base model (model-id 39995) fine-tuned on manually annotated Sámi data
- SamiskOCR_smi_nor (model-id 182005): print M1 base model fine-tuned on manually annotated Sámi and Norwegian data
- SamiskOCR_smi_smipred (model-id 192137): print M1 base model fine-tuned on manually annotated and automatically transcribed Sámi data
- SamiskOCR_alt (model-id 179305): print M1 base model fine-tuned on manually annotated and automatically transcribed Sámi data, and manually annotated Norwegian data

The latter model achieved the best results on our test dataset.

## Tesseract Models
Tesseract is an OCR tool where you can run models locally.
See the [installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html) to install Tesseract.
Once you have installed Tesseract, you can use the .traineddata model files we share here to run OCR on your images.

The following models are available:
- [ub_smi](tesseract_models/ub_smi): a model trained exclusively on manually annotated Sámi data
- [smi](tesseract_models/smi): [the Norwegian base model](https://github.com/tesseract-ocr/tessdata_best/blob/main/nor.traineddata) fine-tuned on manually annotated Sámi data
- [smi_nor](tesseract_models/smi_nor): the Norwegian base model fine-tuned on manually annotated Sámi and Norwegian data
- [smi_pred](tesseract_models/smi_pred): the Norwegian base model fine-tuned on manually annotated and automatically transcribed Sámi data
- [smi_nor_pred](tesseract_models/smi_nor_pred): the Norwegian base model fine-tuned on manually annotated and automatically transcribed Sámi data, and manually annotated Norwegian data
- [synth_base](tesseract_models/synth_base): the Norwegian base model fine-tuned on synthetic* Sámi data
- [sb_smi](tesseract_models/sb_smi): synth_base fine-tuned on manually annotated Sámi data
- [sb_smi_nor_pred](tesseract_models/sb_smi_nor_pred): synth_base fine-tuned on manually annotated and automatically transcribed Sámi data, and manually annotated Norwegian data

The latter model achieved the best results (of the Tesseract models) on our test dataset.

\* synthetic here means that we have real Sámi text, which we have created images of, to resemble scanned text. Unlike the manually transcribed data, which are books and newspapers that are scanned and then manually transcribed.

Tesseract tips:
- You need to move the .traineddata files to the tessdata area on your PC
- To run OCR with the desired model, you must specify the model name with -l, e.g., `tesseract filename.jpg outfile -l smi_nor_pred`. Then `outfile.txt` will contain the text that the model found in the image
- Use --psm 7 if you have an image of a line of text, not a whole page. See more info [here](https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc)

## TrOCR Models
TrOCR is a transformers-based model architecture for text recognition.
You can use the models with the [transformers library](https://huggingface.co/docs/transformers/v4.47.1/en/model_doc/trocr#inference)

The following models are available:
- [trocr_smi](https://huggingface.co/Sprakbanken/trocr_smi): [TrOCR-printed base model](https://huggingface.co/microsoft/trocr-base-printed) fine-tuned on manually annotated Sámi data
- [trocr_smi_nor](https://huggingface.co/Sprakbanken/trocr_smi_nor): TrOCR-printed base model fine-tuned on manually annotated Sámi and Norwegian data
- [trocr_smi_pred](https://huggingface.co/Sprakbanken/trocr_smi_pred): TrOCR-printed base model fine-tuned on manually annotated and automatically transcribed Sámi data
- [trocr_smi_nor_pred](https://huggingface.co/Sprakbanken/trocr_smi_nor_pred): TrOCR-printed base model fine-tuned on manually annotated and automatically transcribed Sámi data, and manually annotated Norwegian data
- [trocr_smi_synth](https://huggingface.co/Sprakbanken/trocr_smi_synth): TrOCR-printed base model fine-tuned on synthetic* Sámi data, and then on manually annotated Sámi data
- [trocr_smi_pred_synth](https://huggingface.co/Sprakbanken/trocr_smi_pred_synth): trocr_smi_synth fine-tuned on manually annotated and automatically transcribed Sámi data
- [trocr_smi_nor_pred_synth](https://huggingface.co/Sprakbanken/trocr_smi_nor_pred_synth): trocr_smi_synth fine-tuned on manually annotated and automatically transcribed Sámi data, and manually annotated Norwegian

trocr_smi_pred_synth is the model that achieved the best results (of the TrOCR models) on our test dataset.

\* synthetic here means that we have real Sámi text, which we have created images of, to resemble scanned text. Unlike the manually transcribed data, which are books and newspapers that are scanned and then manually transcribed.

The models only work with images of lines of text.
If you have images of entire pages of text, you must segment the text into lines first to benefit from this model.

Usage example with python
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
