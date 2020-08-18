import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from core.engine import train_one_epoch, evaluate
from core.coco_utils import get_coco_api_from_dataset
import core.utils
from utils import (
    create_dir,
    get_category_mapping_froom_coco_file,
    Configuration,
)
from transform import get_transforms
from dataset import COCODataset
from model import get_torchvision_maskrcnn

import argparse
import random
import os


class Directories:
    """
    Arranges paths and directories for last_weight_path, best_weight_path, tensorboard_dir
    """

    experiments_dir = "experiments"

    def __init__(self, experiment_name, experiments_dir=experiments_dir):
        self.last_weight_path = os.path.join(
            experiments_dir, experiment_name, "maskrcnn-last.pt"
        )
        self.best_weight_path = os.path.join(
            experiments_dir, experiment_name, "maskrcnn-best.pt"
        )
        self.tensorboard_dir = os.path.join(experiments_dir, experiment_name, "summary")

        last_weight_dir = os.path.dirname(self.last_weight_path)
        best_weight_dir = os.path.dirname(self.best_weight_path)

        create_dir(experiments_dir)
        create_dir(last_weight_dir)
        create_dir(best_weight_dir)


def train(config: dict = None):
    # fix the seed for reproduce results
    SEED = config["SEED"]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)

    # parse config parameters
    DATA_ROOT = config["DATA_ROOT"]
    COCO_PATH = config["COCO_PATH"]
    EXPERIMENT_NAME = config["EXPERIMENT_NAME"]

    OPTIMIZER = config["OPTIMIZER"]
    OPTIMIZER_WEIGHT_DECAY = config["OPTIMIZER_WEIGHT_DECAY"]
    SGD_MOMENTUM = config["SGD_MOMENTUM"]
    ADAM_BETAS = config["ADAM_BETAS"]
    LEARNING_RATE = config["LEARNING_RATE"]
    LEARNING_RATE_STEP_SIZE = config["LEARNING_RATE_STEP_SIZE"]
    LEARNING_RATE_GAMMA = config["LEARNING_RATE_GAMMA"]

    TRAINABLE_BACKBONE_LAYERS = config["TRAINABLE_BACKBONE_LAYERS"]
    RPN_ANCHOR_SIZES = config["RPN_ANCHOR_SIZES"]
    RPN_ANCHOR_ASPECT_RATIOS = config["RPN_ANCHOR_ASPECT_RATIOS"]

    LOG_FREQ = config["LOG_FREQ"]
    TRAIN_SPLIT_RATE = config["TRAIN_SPLIT_RATE"]
    BATCH_SIZE = config["BATCH_SIZE"]
    NUM_EPOCH = config["NUM_EPOCH"]
    DEVICE = config["DEVICE"]
    NUM_WORKERS = config["NUM_WORKERS"]

    # init directories
    directories = Directories(experiment_name=EXPERIMENT_NAME)

    # init tensorboard summary writer
    writer = SummaryWriter(directories.tensorboard_dir)

    # set pytorch device
    device = torch.device(DEVICE)
    if "cuda" in DEVICE and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = torch.device("cpu")

    # use our dataset and defined transformations
    dataset = COCODataset(DATA_ROOT, COCO_PATH, get_transforms(train=True))
    dataset_val = COCODataset(DATA_ROOT, COCO_PATH, get_transforms(train=False))

    # +1 for background class
    num_classes = dataset.num_classes + 1
    config["NUM_CLASSES"] = num_classes

    # add category mappings to config, will be used at prediction
    category_mapping = get_category_mapping_froom_coco_file(COCO_PATH)
    config["CATEGORY_MAPPING"] = category_mapping

    # split the dataset in train and val set
    indices = torch.randperm(len(dataset)).tolist()
    num_train = int(len(indices) * TRAIN_SPLIT_RATE)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    dataset = torch.utils.data.Subset(dataset, train_indices)
    dataset_val = torch.utils.data.Subset(dataset_val, val_indices)

    # define training and val data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=core.utils.collate_fn,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=core.utils.collate_fn,
    )

    # get the model using our helper function
    model = get_torchvision_maskrcnn(
        num_classes=num_classes,
        trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS,
        anchor_sizes=RPN_ANCHOR_SIZES,
        anchor_aspect_ratios=RPN_ANCHOR_ASPECT_RATIOS,
    )

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if OPTIMIZER == "sgd":
        optimizer = OPTIMIZER = torch.optim.SGD(
            params,
            lr=LEARNING_RATE,
            momentum=SGD_MOMENTUM,
            weight_decay=OPTIMIZER_WEIGHT_DECAY
        )
    elif OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=LEARNING_RATE,
            betas=tuple(ADAM_BETAS),
            eps=1e-08,
            weight_decay=OPTIMIZER_WEIGHT_DECAY,
            amsgrad=False,
        )
    else:
        Exception("Invalid OPTIMIZER, try: 'adam' or 'sgd'")

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LEARNING_RATE_STEP_SIZE,
        gamma=LEARNING_RATE_GAMMA
    )

    # create coco index
    print("Creating COCO index...")
    coco_api_train = get_coco_api_from_dataset(data_loader_train.dataset)
    coco_api_val = get_coco_api_from_dataset(data_loader_val.dataset)

    # train it for NUM_EPOCH epochs
    for epoch in range(NUM_EPOCH):
        best_bbox_05095_ap = -1
        # train for one epoch, printing every PRINT_FREQ iterations
        train_one_epoch(
            model,
            optimizer,
            data_loader_train,
            coco_api_train,
            device,
            epoch,
            log_freq=LOG_FREQ,
            writer=writer,
        )
        # update the learning rate
        lr_scheduler.step()
        # get iteration number
        num_images = len(data_loader_train.dataset)
        iter_num = epoch * num_images
        # evaluate on the val dataset
        loss_lists, coco_evaluator = evaluate(
            model=model,
            data_loader=data_loader_val,
            coco_api=coco_api_val,
            device=device,
            iter_num=iter_num,
            writer=writer
        )
        # update best model if it has the best bbox 0.50:0.95 AP
        bbox_05095_ap = coco_evaluator.coco_eval["bbox"].stats[0]
        if bbox_05095_ap > best_bbox_05095_ap:
            model_dict = {"state_dict": model.state_dict(), "config": config}
            torch.save(model_dict, directories.best_weight_path)
            best_bbox_05095_ap = bbox_05095_ap

    # save final model
    model_dict = {"state_dict": model.state_dict(), "config": config}
    torch.save(model_dict, directories.last_weight_path)


if __name__ == "__main__":
    # construct the argument parser
    ap = argparse.ArgumentParser()

    # add the arguments to the parser
    ap.add_argument(
        "config_path", default="configs/config1.yml", help="Path for config file.",
    )
    args = vars(ap.parse_args())

    # read config
    config = Configuration(args["config_path"]).as_dict

    # perform instance segmentation
    train(config)
