import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from core.coco_utils import get_coco_api_from_dataset
from core.coco_eval import CocoEvaluator
import core.utils as utils


def calculate_mean(data_list):
    """
    Calculates mean from given list with float/int elements
    """
    data_mean = sum(data_list) / len(data_list)
    return data_mean


class LossLists:
    def __init__(self):
        self.overall_loss_list = []
        self.loss_classifier_list = []
        self.loss_box_reg_list = []
        self.loss_mask_list = []
        self.loss_objectness_list = []

        self.overall_loss_mean = None
        self.loss_classifier_mean = None
        self.loss_box_reg_mean = None
        self.loss_mask_mean = None
        self.loss_objectness_mean = None

    def append_loss_dict(self, loss_dict):
        """
        Processes given loss_dict and calculates mean loss values
        """

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        self.overall_loss_list.append(losses_reduced)
        self.loss_classifier_list.append(loss_dict_reduced["loss_classifier"])
        self.loss_box_reg_list.append(loss_dict_reduced["loss_box_reg"])
        self.loss_mask_list.append(loss_dict_reduced["loss_mask"])
        self.loss_objectness_list.append(loss_dict_reduced["loss_objectness"])

        self.overall_loss_mean = calculate_mean(self.overall_loss_list)
        self.loss_classifier_mean = calculate_mean(self.loss_classifier_list)
        self.loss_box_reg_mean = calculate_mean(self.loss_box_reg_list)
        self.loss_mask_mean = calculate_mean(self.loss_mask_list)
        self.loss_objectness_mean = calculate_mean(self.loss_objectness_list)

        return loss_dict_reduced, losses_reduced

    def reset(self):
        self.__init__(self)


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    coco_api,
    device,
    epoch,
    log_freq,
    coco_ap_type,
    writer,
):
    # init loss lists instance
    loss_lists = LossLists()

    # init metric logger and model mode
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    # apply warmup schedule for min(1k iter, 1 epoch)
    lr_warmup_schedule = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_warmup_schedule = utils.warmup_lr_scheduler(
            optimizer, warmup_iters, warmup_factor
        )

    iter_num = 0
    num_images = len(data_loader.dataset)
    for images, targets in metric_logger.iterate_over_data_and_log_every(
        data_loader, log_freq, header
    ):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced, losses_reduced = loss_lists.append_loss_dict(loss_dict)

        # log stats for tensorboard
        if iter_num % log_freq == 0:
            writer.add_scalar(
                "overall loss/train",
                loss_lists.overall_loss_mean,
                epoch * num_images + iter_num,
            )
            writer.add_scalar(
                "classifier loss/train",
                loss_lists.loss_classifier_mean,
                epoch * num_images + iter_num,
            )
            writer.add_scalar(
                "box reg loss/train",
                loss_lists.loss_box_reg_mean,
                epoch * num_images + iter_num,
            )
            writer.add_scalar(
                "mask loss/train",
                loss_lists.loss_mask_mean,
                epoch * num_images + iter_num,
            )
            writer.add_scalar(
                "objectness loss/train",
                loss_lists.loss_objectness_mean,
                epoch * num_images + iter_num,
            )
            writer.add_scalar(
                "learning rate/learning rate",
                optimizer.param_groups[0]["lr"],
                epoch * num_images + iter_num,
            )

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_warmup_schedule is not None:
            lr_warmup_schedule.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        iter_num += 1

    # calculate train coco ap
    _ = _calculate_coco_ap(
        model=model,
        data_loader=data_loader,
        coco_api=coco_api,
        device=device,
        iter_num=epoch * num_images + iter_num,
        coco_ap_type=coco_ap_type,
        writer=writer,
        mode="train",
    )


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def _calculate_val_loss(model, data_loader, coco_api, device, iter_num, writer):
    metric_logger = utils.MetricLogger(delimiter="  ")
    model.train()

    # init loss lists instance
    loss_lists = LossLists()

    for images, targets in metric_logger.iterate_over_data_and_log_every(
        data_loader, print_freq=100, header="Val Loss:"
    ):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        _, _ = loss_lists.append_loss_dict(loss_dict)

    # log stats for tensorboard
    writer.add_scalar("overall loss/val", loss_lists.overall_loss_mean, iter_num)
    writer.add_scalar("classifier loss/val", loss_lists.loss_classifier_mean, iter_num)
    writer.add_scalar("box reg loss/val", loss_lists.loss_box_reg_mean, iter_num)
    writer.add_scalar("mask loss/val", loss_lists.loss_mask_mean, iter_num)
    writer.add_scalar("objectness loss/val", loss_lists.loss_objectness_mean, iter_num)

    return loss_lists


def _log_coco_results(writer, mode, category, coco_evaluator, iter_num):
    """
    Logs coco ap results.
    writer: torch.utils.tensorboard.SummaryWriter
    mode: str
        "train", "val"
    category: dict
        {'id': 1, 'name': 'id_card'}
        if given as {'id': -1, 'name': 'overall'}, overall coco ap will be logged instead of category based
    coco_evaluator: core.coco_eval.CocoEvaluator
    iter_num: int
    """
    # get category based coco ap if category id is not -1, else get overall coco ap
    category_id = category["id"]
    category_name = category["name"]

    if category_id is not -1:
        category_index = category["index"]
        coco_evaluator.coco_eval["segm"].params.catIds = [category_id] * category_index
        coco_evaluator.coco_eval["bbox"].params.catIds = [category_id] * category_index

    print("COCO AP for", category_name + ":")
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # log stats for tensorboard
    # AP@0.50:0.95 for bbox
    writer.add_scalar(
        "coco eval bbox/"
        + mode
        + " "
        + category_name
        + ", "
        + " AP@0.50:0.95, all area",
        coco_evaluator.coco_eval["bbox"].stats[0],
        iter_num,
    )
    # AP@0.50 for bbox
    writer.add_scalar(
        "coco eval bbox/" + mode + " " + category_name + ", " + " AP@0.50, all area",
        coco_evaluator.coco_eval["bbox"].stats[1],
        iter_num,
    )
    # AP@0.50:0.95 for mask
    writer.add_scalar(
        "coco eval segm/"
        + mode
        + " "
        + category_name
        + ", "
        + " AP@0.50:0.95, all area",
        coco_evaluator.coco_eval["segm"].stats[0],
        iter_num,
    )
    # AP@0.50 for mask
    writer.add_scalar(
        "coco eval segm/" + mode + " " + category_name + ", " + " AP@0.50, all area",
        coco_evaluator.coco_eval["segm"].stats[1],
        iter_num,
    )


def _calculate_coco_ap(
    model, data_loader, coco_api, device, iter_num, coco_ap_type, writer, mode="val"
):
    """
    Calculates coco ap for given data_loader.
    iter_num, writer and mode is used for logging/tensorboard.
    mode can be specified as "train" or "val".
    """
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco_api, iou_types)

    for images, targets in metric_logger.iterate_over_data_and_log_every(
        data_loader, 100, header=mode + " COCO ap:"
    ):
        images = list(img.to(device) for img in images)

        if device == "cuda":
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # this is to handle both Torchvision Dataset and Subset:
    try:
        categories = data_loader.dataset.dataset.categories
    except:
        categories = data_loader.dataset.categories
    # get all category ids that are present in the current dataset
    present_category_ids = coco_evaluator.coco_eval["bbox"].params.catIds
    present_categories = []
    coco_category_index = 1
    for category in categories:
        if category["id"] in present_category_ids:
            category["index"] = coco_category_index
            present_categories.append(category)
            coco_category_index += 1

    if coco_ap_type == "category_based":
        # log category based coco ap
        for category in present_categories:
            _log_coco_results(writer, mode, category, coco_evaluator, iter_num)
    elif coco_ap_type == "overall":
        # log overall coco ap
        category = {"id": -1, "name": "overall"}
        _log_coco_results(writer, mode, category, coco_evaluator, iter_num)
    else:
        Exception("Invalid COCO_AP_TYPE, try: 'overall', 'category_based'")

    return coco_evaluator


@torch.no_grad()
def evaluate(model, data_loader, coco_api, device, iter_num, coco_ap_type, writer):
    # calculate validation loss
    loss_lists = _calculate_val_loss(
        model, data_loader, coco_api, device, iter_num, writer
    )

    # calculate validation coco ap
    coco_evaluator = _calculate_coco_ap(
        model, data_loader, coco_api, device, iter_num, coco_ap_type, writer, mode="val"
    )

    return loss_lists, coco_evaluator
