import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from core.engine import train_one_epoch, evaluate
from core.coco_utils import get_coco_api_from_dataset
import core.utils
from utils import (
    create_dir,
    get_category_mapping_from_coco_file,
    Configuration,
    save_yaml,
)
from transform import get_transforms
from optimizer import OptimizerFactory
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
        self.experiment_dir = os.path.join(experiments_dir, experiment_name)

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
    DATA_ROOT_VAL = config["DATA_ROOT_VAL"]
    COCO_PATH_VAL = config["COCO_PATH_VAL"]
    EXPERIMENT_NAME = config["EXPERIMENT_NAME"]

    OPTIMIZER_NAME = config["OPTIMIZER_NAME"]
    OPTIMIZER_WEIGHT_DECAY = config["OPTIMIZER_WEIGHT_DECAY"]
    OPTIMIZER_MOMENTUM = config["OPTIMIZER_MOMENTUM"]
    OPTIMIZER_BETAS = config["OPTIMIZER_BETAS"]
    OPTIMIZER_EPS = config["OPTIMIZER_EPS"]
    OPTIMIZER_AMSGRAD = config["OPTIMIZER_AMSGRAD"]
    OPTIMIZER_ADABOUND_GAMMA = config["OPTIMIZER_ADABOUND_GAMMA"]
    OPTIMIZER_ADABOUND_FINAL_LR = config["OPTIMIZER_ADABOUND_FINAL_LR"]
    LEARNING_RATE = config["LEARNING_RATE"]
    LEARNING_RATE_STEP_SIZE = config["LEARNING_RATE_STEP_SIZE"]
    LEARNING_RATE_GAMMA = config["LEARNING_RATE_GAMMA"]

    TRAINABLE_BACKBONE_LAYERS = config["TRAINABLE_BACKBONE_LAYERS"]
    RPN_ANCHOR_SIZES = config["RPN_ANCHOR_SIZES"]
    RPN_ANCHOR_ASPECT_RATIOS = config["RPN_ANCHOR_ASPECT_RATIOS"]
    RPN_PRE_NMS_TOP_N_TRAIN = config["RPN_PRE_NMS_TOP_N_TRAIN"]
    RPN_PRE_NMS_TOP_N_TEST = config["RPN_PRE_NMS_TOP_N_TEST"]
    RPN_POST_NMS_TOP_N_TRAIN = config["RPN_POST_NMS_TOP_N_TRAIN"]
    RPN_POST_NMS_TOP_N_TEST = config["RPN_POST_NMS_TOP_N_TEST"]
    RPN_NMS_THRESH = config["RPN_NMS_THRESH"]
    RPN_FG_IOU_THRESH = config["RPN_FG_IOU_THRESH"]
    RPN_BG_IOU_THRESH = config["RPN_BG_IOU_THRESH"]
    BOX_DETECTIONS_PER_IMAGE = config["BOX_DETECTIONS_PER_IMAGE"]

    LOG_FREQ = config["LOG_FREQ"]
    COCO_AP_TYPE = config["COCO_AP_TYPE"]
    TRAIN_SPLIT_RATE = config["TRAIN_SPLIT_RATE"]
    BATCH_SIZE = config["BATCH_SIZE"]
    NUM_EPOCH = config["NUM_EPOCH"]
    DEVICE = config["DEVICE"]
    NUM_WORKERS = config["NUM_WORKERS"]

    # init directories
    directories = Directories(experiment_name=EXPERIMENT_NAME)

    # copy config file to experiment dir
    yaml_path = os.path.join(directories.experiment_dir, "config.yml")
    save_yaml(config, yaml_path)

    # init tensorboard summary writer
    writer = SummaryWriter(directories.tensorboard_dir)

    # set pytorch device
    device = torch.device(DEVICE)
    if "cuda" in DEVICE and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = torch.device("cpu")

    # use our dataset and defined transformations
    dataset = COCODataset(
        DATA_ROOT, COCO_PATH, get_transforms(config=config, mode="train")
    )
    if COCO_PATH_VAL:
        dataset_val = COCODataset(
            DATA_ROOT_VAL, COCO_PATH_VAL, get_transforms(config=config, mode="val")
        )
    else:
        dataset_val = COCODataset(
            DATA_ROOT, COCO_PATH, get_transforms(config=config, mode="val")
        )

    # +1 for background class
    num_classes = dataset.num_classes + 1
    config["NUM_CLASSES"] = num_classes

    # add category mappings to config, will be used at prediction
    category_mapping = get_category_mapping_from_coco_file(COCO_PATH)
    config["CATEGORY_MAPPING"] = category_mapping

    # split the dataset in train and val set if val path is not defined
    if not COCO_PATH_VAL:
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
        rpn_pre_nms_top_n_train=RPN_PRE_NMS_TOP_N_TRAIN,
        rpn_pre_nms_top_n_test=RPN_PRE_NMS_TOP_N_TEST,
        rpn_post_nms_top_n_train=RPN_POST_NMS_TOP_N_TRAIN,
        rpn_post_nms_top_n_test=RPN_POST_NMS_TOP_N_TEST,
        rpn_nms_thresh=RPN_NMS_THRESH,
        rpn_fg_iou_thresh=RPN_FG_IOU_THRESH,
        rpn_bg_iou_thresh=RPN_BG_IOU_THRESH,
        box_detections_per_img=BOX_DETECTIONS_PER_IMAGE,
        pretrained=True,
    )

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_factory = OptimizerFactory(
        learning_rate=LEARNING_RATE,
        momentum=OPTIMIZER_MOMENTUM,
        weight_decay=OPTIMIZER_WEIGHT_DECAY,
        betas=OPTIMIZER_BETAS,
        eps=OPTIMIZER_EPS,
        amsgrad=OPTIMIZER_AMSGRAD,
        adabound_gamma=OPTIMIZER_ADABOUND_GAMMA,
        adabound_final_lr=OPTIMIZER_ADABOUND_FINAL_LR,
    )
    optimizer = optimizer_factory.get(params, OPTIMIZER_NAME)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LEARNING_RATE_STEP_SIZE, gamma=LEARNING_RATE_GAMMA
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
            model=model,
            optimizer=optimizer,
            data_loader=data_loader_train,
            coco_api=coco_api_train,
            device=device,
            epoch=epoch,
            log_freq=LOG_FREQ,
            coco_ap_type=COCO_AP_TYPE,
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
            coco_ap_type=COCO_AP_TYPE,
            writer=writer,
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
