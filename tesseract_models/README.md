# Forklaring av tesseract-modellene

## Data
- Manuelt annotert og korrigert samisk data
- Manuelt annotert og korrigert norsk data (gt_pix)
- Automatisk transkribert* data (side 30 fra masse forskjellige samiske bøker)

*Med en modell vi har trent i transkribus

### Dataprosessering
- a) fjerne linjene hvor bredden er mindre enn høyden

## Modeller
- smi: bare trent på den manuelt annoterte samiske dataen
- nor_smi: norsk tesseractmodell videretrent på den manuelt annoterte samiske dataen
