# Forklaring av modellene

## Modellnavn
smx er en påfunnet iso-kode for samlingen av flere samiske språk.

## Data
Først hadde vi originaldataen
Så fant vi divvun&giellatekno sin data
Så fikk vi en avis
Og masse automatisk transkriberte side30 fra diverse samiske tekster


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

Unntaket er nor_smx_200.traineddata som er trent på originaldataen med avis + alle side30-dataene. Denne er trent i 200 epoker, ikke iterasjoner.
(brukte iterasjoner i begynnelsen men byttet over til epoker)
