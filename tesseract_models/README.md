# Tesseract models readme

This directory contains a subdirectory for each tesseract model, plus [initial_experiments/](initial_experiments/).

initial_experiments contains the training arguments and plots of model checkpoints from the fine-tuning of the Estoinian, Finnish and Norwegian tessdata-best base models on the GT-Sámi train set.

Each model subdirectory contains the .traineddata model file and a folder with the training details.

## Models overview
[ub_smi](ub_smi) is tesseract model trained from scratch on GT-Sámi (the first row in table 3)

[smi](smi) is [the Norwegian tesseract base model](https://github.com/tesseract-ocr/tessdata_best/blob/main/nor.traineddata) fine-tuned on GT-Sámi  
[smi_nor](smi_nor) is the Norwegian tesseract base model fine-tuned on GT-Sámi and GT-Nor  
[smi_pred](smi_pred) is the Norwegian tesseract base model fine-tuned on GT-Sámi and Pred-Sámi  
[smi_nor_pred](smi_nor_pred) is the Norwegian tesseract base model fine-tuned on GT-Sámi, GT-Nor and Pred-Sámi  

[synth_base](synth_base) is the Norwegian tesseract base model fine-tuned on Synth-Sámi  
[sb_smi](sb_smi) is synth_base fine-tuned on GT-Sámi  
[sb_smi_nor_pred](sb_smi_nor_pred) is synth_base fine-tuned GT-Sámi, GT-Nor and Pred-Sámi  


## Name changes explanation
For some models the names have changed after training, in that case there is an `old_name.txt` file in the training_details folder in each model subdirectory.

In the training script (and therefore in the training arguments json files in the training_details), we have used different names for the dataset configurations than the dataset names in the article. We have combined GT-Sámi (train, val and test splits), GT-Nor and Pred-Sámi in one dataset, and used filter functions to choose which parts of the data to use for training. The gt_pix and page_30 training arguments refer to GT-Nor and Pred-Sámi, respectively.
