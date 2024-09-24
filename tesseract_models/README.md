# Forklaring av tesseract-modellene

## Data
- gt_smi: Manuelt annotert og korrigert samisk data
- gt_nor: Manuelt annotert og korrigert norsk data (gt_pix)
- auto_smi: Automatisk transkribert* data (side 30 fra masse forskjellige samiske bøker)
- synth_smi: Syntetiske bilder av ekte samisk tekst

*Med en modell vi har trent i transkribus

### Dataprosessering
- a) fjerne linjene hvor bredden er mindre enn høyden
- b) fjerne linjene hvor transkripsjonene er kortere enn 5 tegn



## Modeller

### Modeller til artikkelen
- smi: bare trent på den manuelt annoterte samiske dataen
- nor_smi: norsk tesseractmodell videretrent på den manuelt annoterte samiske dataen (gt_smi)
- nor_smi_nor: norsk tesseractmodell videretrent på den manuelt annoterte dataen (gt_smi + gt_nor)
- nor_smi_smi: norsk tesseractmodell videretrent på den samiske dataen (gt_smi, auto_smi)


### Basemodell-eksperiment
Trent én runde med tesseract på forskjellige basemodeller med vår data.

- nor_smi1: CER: 5.03% WER: 11.97%
- est_smi1: CER: 5.15% WER: 12.29%
- fin_smi1: CER: 5.94% WER: 16.15%
