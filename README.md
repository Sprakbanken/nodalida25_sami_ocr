# Samisk OCR

Dette repoet inneholder modeller trent på samisk OCR.
Foreløpig er det kun tesseractmodeller her.

Se dokumentet [tesseract howto](tesseract_howto.md) for hvordan du kan bruke (og trene!) disse modellene

## Treningsdata
Vi har transkribert vår egen data, som er en blanding av flere forskjellige samiske språk (les mer [her](https://bibno-my.sharepoint.com/:w:/r/personal/marie_rosok_nb_no/Documents/Chatfiler%20for%20Microsoft%20Teams/SamiskOCR-notat.docx?d=wc077d3c74c4a4bb8ab16a9a4dcb5b45d&csf=1&web=1&e=7ZvMmv))

Vi har også brukt dataen som Divvun & Giellatekno har på sin [github](https://github.com/divvungiellatekno/tesstrain/tree/main/training-data), som er nordsamisk data (sme).

## Modeller
Modellene er navngitt eller eventuelle basemodeller og antall epoker de er trent.
Eksempel:
```
est_smx_2000.traineddata
```
est: modellen er basert på den estiske basemodellen som finnes i [tessdata_best-repoet](https://github.com/tesseract-ocr/tessdata_best).
smx: den er finetuned på vårt transkriberte datasett
2000: den er trent i 2000 epoker

smx er en liksom-iso-kode for samlingen av flere samiske språk.
Når det står smx2 i modellnavnet betyr det vår transkriberte data + Divvun & Giellatekno sin data

## Resultater
TBA
