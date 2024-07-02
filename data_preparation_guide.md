# Data preparation
This is a guide on how to prepare line-level data for training OCR-models from an export job from Transkribus.  
Assumes the Transkribus data contains alto-xml files with bounding boxes and transcriptions.

## Step 1: extract line-level images
Clone the hugin-munin-ordbilder repository and install the package:  
```
git clone git@github.com:Sprakbanken/hugin-munin-ordbilder.git
cd hugin-munin-ordbilder
git checkout uten_author # todo: remove this after merge
pip install .
cd ..
``` 
Run the script [create_unadjusted_dataset_no_author.py](https://github.com/Sprakbanken/hugin-munin-ordbilder/blob/uten_author/scripts/create_unadjusted_dataset_no_author.py):  
```
python3 hugin-munin-ordbilder/scripts/create_unadjusted_dataset_no_author.py [unzipped_transkribus_export_job] [output_directory]  
```

## Step 2: rearrange images and texts
Create a directory with pairs of .tif line images and .gt.txt line transcriptions.  
See [rearrange_data.ipynb](rearrange_data.ipynb)
