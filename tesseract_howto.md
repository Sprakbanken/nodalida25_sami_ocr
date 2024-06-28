

# Inference with tesseract 

## Step1: install tesseract 
Follow instructions in repo

## Step2: run OCR-model:
Make sure you have the `.traineddata` file of the model you want to use for inference in your tessdata directory  

Run this command
`tesseract [imagefile] [outputfile] -l [lang]`  
Replace `[imagefile]` with the path to an image  
Replace `[outputfile]` with the path to write the detected text (it will generate `[outputfile].txt`)
Where `[lang]` is whatever comes before .traineddata of the model you want to use for inference


# Training models with tesstrain 

## Step1: install tesstrain and tesseract 
Følg slik det står i dokumentasjonen til tesstrain repoet og tesseract repoet (todo: fyll inn her når jeg har reprodusert)

## Step2: prepare data for training
Make sure the data you want to use for fine-tuning is in `tesstrain/data/[model_name]-ground-truth` where `[model_name]` is the same as the MODEL_NAME parameter you pass to the training function.  
In `tesstrain/data/[model_name]-ground-truth` you should have pairs of `[filename].tif` and `[filename].gt.txt` files, where the .tif file is a line image, and the .gt.txt file is the text in that line.  

## Step3a: fine-tune an existing tesseract model

### First
Set the TESSDATA_PREFIX environment variable to point to your tessdata directory:  
`export TESSDATA_PREFIX=[your-tessdata-directory]`  
(Replace `[your-tessdata-directory]` with the path to your tessdata directory, e.g `/usr/local/share/tessdata/`)  
You can run this command in the terminal and it will last for your session, or you can paste it to your ~/.bashrc file and it will always be set (remember to `source ~/.bashrc` after changing the file)  
Check that is has been set with `echo $TESSDATA_PREFIX` in the terminal

### Then
Make sure you have the `.traineddata` file of the model you want to continue training from in your tessdata directory.   

option 1: 
You can download a language base model from the tesseract repo like this:  
`wget https://github.com/tesseract-ocr/tessdata_best/raw/main/[lang].traineddata -P $TESSDATA_PREFIX`  
(Replace `[lang]` with the 3-charcter iso-langcode (and make sure a model for your language actually exists by browsing [the repo](https://github.com/tesseract-ocr/tessdata_best/)). Depending on access rights to the tessdata directory you might have to add `sudo` before `wget`. )

option2:
Continue training from your own model. Copy or move the `.traineddata` file to the tessdata directory

### Finally
Run this command in the root of the tesstrain directory: 
`make training MODEL_NAME=[model_name] START_MODEL=[lang] TESSDATA=$TESSDATA_PREFIX`
Where `[lang]` is whatever comes before .traineddata of the model you want to continue training from. 


## Step3b: train a tesseract model from scratch

