# Augmented MaskRCNN
This repo let's you **easily fine tune pretrained MaskRCNN model with 64 fast image augmentation types using your custom data/annotations, then apply prediction** based on the trained model. Training and inference works both on **Windows & Linux**.
- [albumentations](https://github.com/albumentations-team/albumentations) is integrated for image augmentation which is [much faster](https://github.com/albumentations-team/albumentations#benchmarking-results) than [imgaug](https://github.com/aleju/imgaug) and **supports 64 augmentation types for images, bounding boxes and masks**
- Pretrained **resnet50 + feature pyramid** weights on COCO is downloaded upon training
- **COCO evaluation** is performed after each epoch, during training

## Installation
Manually install Miniconda (Python3) for your OS:
https://docs.conda.io/en/latest/miniconda.html

Install Miniconda (Python3) by bash script on Linux:
```console
~$ sudo apt update --yes
~$ sudo apt upgrade --yes
~$ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
~$ bash ~/miniconda.sh -b -p ~/miniconda 
~$ rm ~/miniconda.sh
```

Inside the base project directory, open up a terminal/anaconda promt window and create environment:
```console
~$ conda env create -f environment.yml
```

- After environment setup, activate environment and run the tests to see if everything is ready:
```console
conda activate augmented-maskrcnn
python -m unittest
```

## Usage
- In the base project directory, open up a terminal/anaconda promt window, and activate environment:
```console
~$ conda activate augmented-maskrcnn
```

- Edit config.py for your needs

- Perform training:
```console
~$ python train.py
```

- Perform prediction for image "test/test_files/CA/CA01_01.tif" using model "artifacts/maskrcnn-best.pt":
```console
~$ python predcit.py test/test_files/CA/CA01_01.tif artifacts/maskrcnn-best.pt
```
