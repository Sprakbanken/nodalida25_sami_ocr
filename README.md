# Samisk OCR
Dette repoet inneholder modeller trent på samisk OCR.

## Treningsdata
Vi har transkribert vår egen data, som er en blanding av flere forskjellige samiske språk (les mer [her](https://bibno-my.sharepoint.com/:w:/r/personal/marie_rosok_nb_no/Documents/Chatfiler%20for%20Microsoft%20Teams/SamiskOCR-notat.docx?d=wc077d3c74c4a4bb8ab16a9a4dcb5b45d&csf=1&web=1&e=7ZvMmv))

I tillegg har vi automatisk transkribert side 30 fra en rekke samiske bøker med en modell vi har trent i transkribus. Dette for å få litt volum på datamengden, om enn noe lavere kvalitet.

## Modeller

### Tesseract 
Modellene ligger i [tesseract_models](tesseract_models)
Les mer om modellene i [README-fila](tesseract_models/README.md)

Se dokumentet [tesseract howto](tesseract_howto.md) for hvordan du kan bruke (og trene!) disse modellene

### TrOCR
Modellene ligger på [Språkbankens sider på huggingface](https://huggingface.co/Sprakbanken)

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
python3 -m <skriptnavn>
```
eller, med pdm
```
pdm run python -m <skriptnavn>
```
Legg til `--help` for å få mer informasjon om argumenter
