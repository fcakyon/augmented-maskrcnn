import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from core.engine import train_one_epoch, evaluate
import core.utils
import torch
import random
import os

from dataset import COCODataset
from config import configurations

from albumentations import (
    BboxParams,
    HorizontalFlip,
    #VerticalFlip,
    RandomRotate90,
    RandomBrightness,
    MotionBlur,
    Compose,
    Normalize
)


def get_model_instance_segmentation(num_classes: int):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train: bool) -> Compose:
    transforms = Compose(
            [Normalize()],
            bbox_params=BboxParams(format='pascal_voc',
                                   min_area=0.,
                                   min_visibility=0.,
                                   label_fields=['category_id']))

    # compose train transforms
    if train:
        transforms = Compose(
                [Normalize(),
                 HorizontalFlip(),
                 RandomRotate90(),
                 RandomBrightness(),
                 MotionBlur()],
                bbox_params=BboxParams(format='pascal_voc',
                                       min_area=0.,
                                       min_visibility=0.,
                                       label_fields=['category_id']))
    return transforms


def train(config=None):
    if config is None:
        cfg = configurations[1]
    else:
        cfg = config

    # fix the seed for reproduce results
    SEED = cfg['SEED']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)

    # parse config parameters
    DATA_ROOT = cfg["DATA_ROOT"]
    COCO_PATH = cfg["COCO_PATH"]

    ARTIFACT_DIR = cfg["ARTIFACT_DIR"]
    EXPERIMENT_NAME = cfg["EXPERIMENT_NAME"]

    BATCH_SIZE = cfg["BATCH_SIZE"]
    NUM_EPOCH = cfg["NUM_EPOCH"]

    DEVICE = cfg["DEVICE"]
    NUM_WORKERS = cfg["NUM_WORKERS"]

    # train on the GPU or on the CPU, if a GPU is not available
    device = DEVICE

    # use our dataset and defined transformations
    dataset = COCODataset(DATA_ROOT, COCO_PATH,
                          get_transform(train=True))
    dataset_test = COCODataset(DATA_ROOT, COCO_PATH,
                               get_transform(train=False))

    # our dataset has two classes only - background and id card
    num_classes = dataset.num_objects + 1

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and test data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=core.utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=core.utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for NUM_EPOCH epochs
    for epoch in range(NUM_EPOCH):
        best_bbox_05095_ap = -1
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch,
                        print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        coco_evaluator = evaluate(model, data_loader_test, device=device)
        # update best model if it has the best bbox 0.50:0.95 AP
        bbox_05095_ap = coco_evaluator.coco_eval["bbox"].stats[0]
        if bbox_05095_ap > best_bbox_05095_ap:
            save_path = os.path.join(ARTIFACT_DIR,
                                     EXPERIMENT_NAME + '-best.pt')
            model_dict = {"state_dict": model.state_dict(), "cfg": cfg}
            torch.save(model_dict, save_path)
            best_bbox_05095_ap = bbox_05095_ap

    # save final model
    save_path = os.path.join(ARTIFACT_DIR,
                             EXPERIMENT_NAME + '-last.pt')
    model_dict = {"state_dict": model.state_dict(), "cfg": cfg}
    torch.save(model_dict, save_path)


if __name__ == "__main__":
    train()
