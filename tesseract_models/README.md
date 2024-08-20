# Forklaring av modellene

## Modellnavn
smx er en påfunnet iso-kode for samlingen av flere samiske språk.

## Data
Først hadde vi originaldataen
Så fant vi divvun&giellatekno sin data
Så fikk vi en avis
Og masse automatisk transkriberte side30 fra diverse samiske tekster

### Dataprosessering
- a) fjerne linjene hvor bredden er mindre enn høyden
- b) rotere linjene hvor bredden er mindre enn halvparten av høyden
- c) fjerne linjene hvor transkripsjonen er kortere enn x tegn
- d) erstatte forekomster av em-dash med en-dash

## Modeller
nor_smx_200.traineddata
- videretrent fra nor.traineddata fra tessdata_best
- trent på originaldataen med avis + alle side30-dataene
- trent i 200 epoker

nor_smx_201.traineddata
- videretrent fra nor_smx_200.traineddata
- trent på originaldataen + dataprosessering a, b, c (x=5) og d
- trent i 1 epoke

nor_smx_205.traineddata
- videretrent fra nor_smx_205.traineddata
- med 10x lavere learning rate enn default (0.0002)
- 5 epoker, men samme data

nor_smx_206.traineddata
- videretrent fra nor_smx_205.traineddata
- trent på originaldataen + dataprosessering a, b, c (x=5) og d
- trent i 1 epoke

nor_smx_train_rotate_remove.traineddata
- trent på originaldataen med avis, men etter dataprossesseringssteg a og b
- 100 epoker



## Modellnavn i gamle_modeller
Modellene er navngitt etter eventuelle basemodeller og antall iterasjoner de er trent.
Eksempel:
```
est_smx_20000.traineddata
```
est: modellen er basert på den estiske basemodellen som finnes i [tessdata_best-repoet](https://github.com/tesseract-ocr/tessdata_best).
smx: den er finetuned på vårt transkriberte datasett
20000: den er trent i 20 000 iterasjoner

Modellene i `gamle_modeller/` er trent på originaldataen uten avis hvis de starter med smx_, og originaldataen uten avis + divvun&giellatekno sin data hvis de starter med smx2_
