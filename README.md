# Augmented MaskRCNN

[![CI](https://github.com/fcakyon/augmented-maskrcnn/workflows/CI/badge.svg)](https://github.com/fcakyon/augmented-maskrcnn/actions?query=event%3Apush+branch%3Amaster+is%3Acompleted+workflow%3ACI)

This repo lets you **easily fine tune pretrained MaskRCNN model with 64 fast image augmentation types using your custom data/annotations, then apply prediction** based on the trained model. Training and inference works both on **Windows & Linux**.

- [torchvision](https://github.com/pytorch/vision) is integrated for MaskRCNN training which provides faster convergence with [negative sample support](https://github.com/pytorch/vision/releases/tag/v0.6.0)
- [albumentations](https://github.com/albumentations-team/albumentations) is integrated for image augmentation which is [much faster](https://github.com/albumentations-team/albumentations#benchmarking-results) than [imgaug](https://github.com/aleju/imgaug) and **supports 64 augmentation types for images, bounding boxes and masks**
- [torch-optimizer](https://github.com/jettify/pytorch-optimizer) is integrated to support [AdaBound](https://arxiv.org/abs/1902.09843), [Lamb](https://arxiv.org/abs/1904.00962), [RAdam](https://arxiv.org/abs/1908.03265) optimizers.
- [tensorboard](https://github.com/tensorflow/tensorboard) is integrated for visualizing the training/validation losses, category based training/validation coco ap results and iteration based learning rate changes
- Pretrained **resnet50 + feature pyramid** weights on COCO is downloaded upon training
- **COCO evaluation** is performed after each epoch, for training and validation sets for each category

## Installation

- Manually install Miniconda (Python3) for your OS:
https://docs.conda.io/en/latest/miniconda.html

Or install Miniconda (Python3) by bash script on Linux:

```console
sudo apt update --yes
sudo apt upgrade --yes
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda
rm ~/miniconda.sh
```

- Inside the base project directory, open up a terminal/anaconda promt window and create environment:

```console
conda env create -f environment.yml
```

- After environment setup, activate environment and run the tests to see if everything is ready:

```console
conda activate augmented-maskrcnn
python -m unittest
```

## Usage

- In the base project directory, open up a terminal/anaconda promt window, and activate environment:

```console
conda activate augmented-maskrcnn
```

- Create a config yml file similar to [default_config.yml](configs/default_config.yml) for your needs

- Perform training by giving the config path as argument:

```console
python train.py configs/default_config.yml
```

- Visualize realtime training/validation losses and accuracies via tensorboard from http://localhost:6006 in your browser:

```console
tensorboard --logdir=experiments
```

- Perform prediction for image "CA01_01.tif" using model "artifacts/maskrcnn-best.pt":

```console
python predict.py CA01_01.tif artifacts/maskrcnn-best.pt
```
