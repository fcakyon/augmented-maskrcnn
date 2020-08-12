import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.tensorboard import SummaryWriter

from core.engine import train_one_epoch, evaluate
import core.utils
import torch
import random
import os

from utils import create_dir, get_category_mapping_froom_coco_file
from dataset import COCODataset
from config import configurations

from albumentations.core.composition import BboxParams, Compose

from albumentations.augmentations.transforms import (
    LongestMaxSize,
    PadIfNeeded,
    RandomSizedBBoxSafeCrop,
    ShiftScaleRotate,
    RandomRotate90,
    HorizontalFlip,
    RandomBrightnessContrast,
    RandomGamma,
    HueSaturationValue,
    MotionBlur,
    JpegCompression,
    Normalize,
)


class Directories:
    """
    Arranges paths and directories for last_weight_path, best_weight_path, tensorboard_dir
    """
    experiments_dir = "experiments"

    def __init__(self, experiment_name, experiments_dir=experiments_dir):
        self.last_weight_path = os.path.join(experiments_dir, experiment_name, "maskrcnn-last.pt")
        self.best_weight_path = os.path.join(experiments_dir, experiment_name, "maskrcnn-best.pt")
        self.tensorboard_dir = os.path.join(experiments_dir, experiment_name, "summary")

        last_weight_dir = os.path.dirname(self.last_weight_path)
        best_weight_dir = os.path.dirname(self.best_weight_path)

        create_dir(experiments_dir)
        create_dir(last_weight_dir)
        create_dir(best_weight_dir)


def get_model_instance_segmentation(num_classes: int = 91, trainable_backbone_layers: int = 3):
    # load an instance segmentation model pre-trained on COCO
    anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        trainable_backbone_layers=trainable_backbone_layers,
        pretrained=True,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_fg_iou_thresh=0.5
    )
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def get_transform(train: bool) -> Compose:
    transforms = Compose(
        [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
        bbox_params=BboxParams(
            format="pascal_voc",
            min_area=0.0,
            min_visibility=0.0,
            label_fields=["category_id"],
        ),
    )

    # compose train transforms
    if train:
        transforms = Compose(
            [
                LongestMaxSize(max_size=768, p=1),
                PadIfNeeded(min_height=768, min_width=768, border_mode=0, p=1),
                RandomSizedBBoxSafeCrop(height=768, width=768, p=0.5),
                HorizontalFlip(p=0.5),
                RandomRotate90(p=0),
                RandomBrightnessContrast(p=0.3),
                RandomGamma(p=0),
                HueSaturationValue(p=0),
                MotionBlur(p=0),
                JpegCompression(quality_lower=20, quality_upper=95, p=0),
                Normalize(
                    max_pixel_value=255.0,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    p=1,
                ),
            ],
            bbox_params=BboxParams(
                format="pascal_voc",
                min_area=0.0,
                min_visibility=0.0,
                label_fields=["category_id"],
            ),
        )
    return transforms


def train(config=None):
    if config is None:
        cfg = configurations[1]
    else:
        cfg = config

    # fix the seed for reproduce results
    SEED = cfg["SEED"]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)

    # parse config parameters
    DATA_ROOT = cfg["DATA_ROOT"]
    COCO_PATH = cfg["COCO_PATH"]

    EXPERIMENT_NAME = cfg["EXPERIMENT_NAME"]
    TRAINABLE_BACKBONE_LAYERS = cfg["TRAINABLE_BACKBONE_LAYERS"]
    PRINT_FREQ = cfg["PRINT_FREQ"]
    OPTIMIZER = cfg["OPTIMIZER"]
    LEARNING_RATE = cfg["LEARNING_RATE"]

    BATCH_SIZE = cfg["BATCH_SIZE"]
    NUM_EPOCH = cfg["NUM_EPOCH"]

    DEVICE = cfg["DEVICE"]
    NUM_WORKERS = cfg["NUM_WORKERS"]

    # init directories
    directories = Directories(experiment_name=EXPERIMENT_NAME)

    # init tensorboard summary writer
    writer = SummaryWriter(directories.tensorboard_dir)

    # train on the GPU or on the CPU, if a GPU is not available
    device = DEVICE

    # use our dataset and defined transformations
    dataset = COCODataset(DATA_ROOT, COCO_PATH, get_transform(train=True))
    dataset_test = COCODataset(DATA_ROOT, COCO_PATH, get_transform(train=False))

    # our dataset has two classes only - background and id card
    num_classes = dataset.num_classes + 1
    cfg["NUM_CLASSES"] = num_classes

    # add category mappings to cfg, will be used at prediction
    category_mapping = get_category_mapping_froom_coco_file(COCO_PATH)
    cfg["CATEGORY_MAPPING"] = category_mapping

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    num_train = int(len(indices)*0.8)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    dataset = torch.utils.data.Subset(dataset, train_indices)
    dataset_test = torch.utils.data.Subset(dataset_test, test_indices)

    # define training and test data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=core.utils.collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=core.utils.collate_fn,
    )

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes, TRAINABLE_BACKBONE_LAYERS)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if OPTIMIZER == "sgd":
        optimizer = OPTIMIZER=torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    elif OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005, amsgrad=False)
    else:
        Exception("Invalid OPTIMIZER, try: 'adam' or 'sgd'")
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for NUM_EPOCH epochs
    for epoch in range(NUM_EPOCH):
        best_bbox_05095_ap = -1
        # train for one epoch, printing every PRINT_FREQ iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=PRINT_FREQ, writer=writer)
        # update the learning rate
        lr_scheduler.step()
        # get iteration number
        num_images = len(data_loader.dataset)
        iter_num = epoch*num_images
        # evaluate on the val dataset
        loss_lists, coco_evaluator = evaluate(model, data_loader_test, device=device, iter_num=iter_num, writer=writer)
        # update best model if it has the best bbox 0.50:0.95 AP
        bbox_05095_ap = coco_evaluator.coco_eval["bbox"].stats[0]
        if bbox_05095_ap > best_bbox_05095_ap:
            model_dict = {"state_dict": model.state_dict(), "cfg": cfg}
            torch.save(model_dict, directories.best_weight_path)
            best_bbox_05095_ap = bbox_05095_ap

    # save final model
    model_dict = {"state_dict": model.state_dict(), "cfg": cfg}
    torch.save(model_dict, directories.last_weight_path)


if __name__ == "__main__":
    train()
