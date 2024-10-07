# Forklaring av tesseract-modellene

## Data
- gt_smi: Manuelt annotert og korrigert samisk data
- gt_nor: Manuelt annotert og korrigert norsk data (tidligere gt_pix)
- pred_smi: Automatisk transkribert* samisk data (tidligere page_30/side_30)
- synth_smi: Syntetiske bilder av ekte samisk tekst

*Med en modell vi har trent i transkribus

### Dataprosessering
- a) fjerne linjene hvor bredden er mindre enn høyden
- b) fjerne linjene hvor transkripsjonene er kortere enn 5 tegn

a) gjøres for all data, b) gjøres kun for pred_smi

## Modeller

- ub_smi: tesseractmodell trent på den manuelt annoterte samiske dataen
- smi: norsk tesseractmodell videretrent på den manuelt annoterte samiske dataen (gt_smi)
- smi_nor: norsk tesseractmodell videretrent på den manuelt annoterte dataen (gt_smi + gt_nor)
- smi_pred: norsk tesseractmodell videretrent på den samiske dataen (gt_smi + pred_smi)
- smi_nor_pred:  norsk tesseractmodell videretrent på gt_smi + pred_smi + gt_nor

- ub_smi: CER: 5.03% WER: 11.97%
- smi: CER: 5.03% WER: 11.97%
- smi_nor: CER: 5.03% WER: 11.97%
- smi_pred: CER: 5.03% WER: 11.97%
- smi_nor_pred: CER: 5.03% WER: 11.97%

Modellene hadde andre navn under trening, dette ligger som old_name.txt i hver modellmappe

### Basemodell-eksperiment
Trent én runde med tesseract på forskjellige basemodeller med vår data.

- nor_smi1: CER: 5.03% WER: 11.97%
- est_smi1: CER: 5.15% WER: 12.29%
- fin_smi1: CER: 5.94% WER: 16.15%
