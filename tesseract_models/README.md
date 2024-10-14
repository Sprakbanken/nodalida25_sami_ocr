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
Merk: Noen av modellene hadde andre navn under trening, dette ligger som old_name.txt i hver modellmappe

### ub_smi
tesseractmodell trent på den manuelt annoterte samiske dataen
Resultater på valideringssett:
CER: 7.93% WER: 24.7%

### nor
norsk tesseract_modell (fra tesseract sin github)
Resultater på valideringssett:
CER: 13.77% WER: 44.04%

### smi
norsk tesseractmodell videretrent på den manuelt annoterte samiske dataen (gt_smi)
Resultater på valideringssett:
CER: 4.59% WER: 9.84%

### smi_nor
norsk tesseractmodell videretrent på den manuelt annoterte dataen (gt_smi + gt_nor)
Resultater på valideringssett:
CER: 4.91% WER: 11.39%

### smi_pred
norsk tesseractmodell videretrent på den samiske dataen (gt_smi + pred_smi)
Resultater på valideringssett:
CER: 4.42% WER: 8.17%

### smi_nor_pred
norsk tesseractmodell videretrent på gt_smi + pred_smi + gt_nor
Resultater på valideringssett:
CER: 4.4% WER: 7.96%

### synth_base
norsk tesseractmodell videretrent på synth_smi
Resultater på valideringssett:
CER: 5.56% WER: 12.88

### sb_smi
synth_base videretrent på den manuelt annoterte samiske dataen (gt_smi)
Resultater på valideringssett:
4.33 WER: 8.78%

### sb_smi_nor_pred
synth_base videretrent på gt_smi + pred_smi + gt_nor
Resultater på valideringssett:
CER: 4.36 WER 7.7%


### Basemodell-eksperiment
Trent én runde med tesseract på forskjellige basemodeller med vår data.

- nor_smi1: CER: 5.03% WER: 11.97%
- est_smi1: CER: 5.15% WER: 12.29%
- fin_smi1: CER: 5.94% WER: 16.15%
