# Samisk OCR

Dette repoet inneholder modeller trent på samisk OCR.
Foreløpig er det kun tesseractmodeller her.

Se dokumentet [tesseract howto](tesseract_howto.md) for hvordan du kan bruke (og trene!) disse modellene

## Treningsdata
Vi har transkribert vår egen data, som er en blanding av flere forskjellige samiske språk (les mer [her](https://bibno-my.sharepoint.com/:w:/r/personal/marie_rosok_nb_no/Documents/Chatfiler%20for%20Microsoft%20Teams/SamiskOCR-notat.docx?d=wc077d3c74c4a4bb8ab16a9a4dcb5b45d&csf=1&web=1&e=7ZvMmv))

Vi har også brukt dataen som Divvun & Giellatekno har på sin [github](https://github.com/divvungiellatekno/tesstrain/tree/main/training-data), som er nordsamisk data (sme).

## Modeller
Modellene ligger i [tesseract_models](tesseract_models)
Les mer om modellene i [README-fila](tesseract_models/README.md)

## Resultater
TBA


## Installasjon
Du kan lage et virtuelt pythonmiljø og installere slik:
(fordrer at du har riktig versjon av python og at venv er installert)
```
python3 -m venv <navn-på-miljøet>
. <navn-på-miljøet>/bin/activate
pip install .
```

Eller hvis du liker pdm:
```
python3 -m pip install pdm # installer pdm hvis du ikke har
pdm install
```

## Kodemoduler
Alle skriptene kan kjøres med
```
python -m <skriptnavn>
```
eller, med pdm
```
pdm run python -m <skriptnavn>
```
Legg til `--help` for å få mer informasjon om argumenter

[transkribus_export_to_line_data](src/transkribus_export_to_line_data.py) tar en transkribus-export med bilder og transkripsjoner og gjør det om til bildefiler og tekstfiler på linjenivå (brukes til trening)

[tesseract_transcribe](src/tesseract_transcribe.py) transkribere alle bildene i en mappe med en valgfri tesseract-modell og skriver resultatene i en .csv-fil

[transkribus_export_to_prediction_file](src/transkribus_export_to_prediction_file.py) tar en transkribus-export med bilder og transkripsjoner og lager en fil med samme struktur som tesseract transcribe

[evaluate_predictions](src/evaluate_predictions.py) tar inn en .csv-fil (output fra de to over) og regner ut WER og CER på samling, side og linjenivå

[find_bad_boxes](src/find_bad_boxes.py) er en hjelpefunksjon for å finne tilfeller der boksene
