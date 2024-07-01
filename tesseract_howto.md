

# Inference with Tesseract 

## Step1: install Tesseract 5
Follow install instructions in [the official documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html)  
Confirm installation with `tesseract --version`

## Step2: prepare model:
Run `tesseract --list-langs` to see the available models  
If you dont see the model you want to use, copy or move the `.traineddata` file of the model the tessdata directory.  
Your tessdata directory may be something like `/usr/local/share/tessdata/` or `/usr/share/tesseract-ocr/5/tessdata/` depending on your installation.  

Example:  
`cp tesseract_models/nor_smx_15000.traineddata [path_to_tessdata]`  
You may have to put sudo before cp to write to your tessdata repository

## Step3: run OCR-model
Run this command
`tesseract [imagefile] [outputfile] -l [model_name]`  
Replace `[imagefile]` with the path to an image  
Replace `[outputfile]` with the path to write the detected text (it will generate `[outputfile].txt`)
Where `[model_name]` is whatever comes before .traineddata of the model you want to use for inference

Example:  
`tesseract example_image.tif example_image.out -l nor_smx_15000`  
In `example_image.out.txt` you will find the output of the model

See [tesseract_transcribe.ipynb](tesseract_transcribe.ipynb) for how to transcribe multiple images with pytesseract

# Training models with tesstrain 

## Step1: install tesstrain and tesseract dev tools
Clone the [tesstrain repository](https://github.com/tesseract-ocr/tesstrain) and follow installation instructions in the README file. (If you've already installed tesseract you might be good to go after just cloning the repo)

## Step2: prepare data for training
Make sure the data you want to use for fine-tuning is in `tesstrain/data/[model_name]-ground-truth` where `[model_name]` is the same as the MODEL_NAME parameter you pass to the training function.  
In `tesstrain/data/[model_name]-ground-truth` you should have pairs of `[filename].tif` and `[filename].gt.txt` files, where the .tif file is a line image, and the .gt.txt file is the text in that line.  
See [sample_data](sample_data/) for an example pair  

## Step3: train!

### Option A: train a tesseract model from scratch
Navigate to the root of the tesstrain directory.  
Run this command:  
`make training MODEL_NAME=[model_name]`  
You can optionally add other training parameters (such as `MAX_ITERATIONS`) as described in the tesstrain README.

### Option B: fine-tune/continue training on an existing tesseract

#### First
Set the `TESSDATA_PREFIX` environment variable to point to your tessdata directory:  
`export TESSDATA_PREFIX=[your-tessdata-directory]`  
(Replace `[your-tessdata-directory]` with the path to your tessdata directory, e.g `/usr/local/share/tessdata/`)  
You can run this command in the terminal and it will last for your session, or you can paste it to your ~/.bashrc file and it will always be set (remember to `source ~/.bashrc` after changing the file)  
Check that is has been set with `echo $TESSDATA_PREFIX` in the terminal

#### Then
Make sure you have the `.traineddata` file of the model you want to continue training from in your tessdata directory.   

option 1: 
You can download a language base model from the tesseract repo like this:  
`wget https://github.com/tesseract-ocr/tessdata_best/raw/main/[lang].traineddata -P $TESSDATA_PREFIX`  
(Replace `[lang]` with the 3-charcter iso-langcode (and make sure a model for your language actually exists by browsing [the repo](https://github.com/tesseract-ocr/tessdata_best/)). Depending on access rights to the tessdata directory you might have to add `sudo` before `wget`. )

option2:
Continue training from your own model. Copy the `.traineddata` file to the tessdata directory  
`cp [path_to_source_model].traineddata $TESSDATA_PREFIX`  
Depending on access rights to the tessdata directory you might have to add `sudo` before `cp`.

#### Finally
Navigate to the root of the tesstrain directory.  

Run this command:  
`make training MODEL_NAME=[model_name] START_MODEL=[lang] TESSDATA=$TESSDATA_PREFIX`  
Where `[lang]` is whatever comes before .traineddata of the model you want to continue training from.  
You can optionally add other training parameters (such as `MAX_ITERATIONS`) as described in the tesstrain README.

