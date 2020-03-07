import os
import cv2
import json
import torch
import random
import argparse
import numpy as np
from albumentations import (
    Compose,
    Normalize
)
from utils import create_dir
import matplotlib.pyplot as plt
from train import get_model_instance_segmentation


def get_transform() -> Compose:
    transforms = Compose([Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    return transforms


def to_float_tensor(img):
    # Converts numpy images to pytorch format
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


def random_colour_masks(image: np.array):
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255],
               [255, 255, 0], [255, 0, 255], [80, 70, 180], [250, 80, 190],
               [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    (r[image == 1],
     g[image == 1],
     b[image == 1]) = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def get_prediction(image_path: str, model,
                   cfg: dict, threshold: float = 0.5,
                   verbose: int = 1) -> (list, list, list):
    # load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # apply transform
    transforms = get_transform()
    augmented = transforms(image=image)
    image = augmented["image"]
    # convert to tensor
    image = to_float_tensor(image).unsqueeze(0)

    # get prediction
    model.eval()
    pred = model(image)

    # get coco categories
    COCO_PATH = cfg["COCO_PATH"]
    with open(COCO_PATH) as json_file:
        coco_dict = json.load(json_file)
        COCO_CATEGORIES = coco_dict["categories"]
    INSTANCE_CATEGORY_NAMES = {
            COCO_CATEGORY["id"]: COCO_CATEGORY["name"] for
            COCO_CATEGORY in COCO_CATEGORIES
            }

    # get predictions with above threshold prediction scores
    pred_score = list(pred[0]['scores'].detach().numpy())
    num_predictions_above_threshold = sum([1 for x in pred_score
                                           if x > threshold])
    pred_num = num_predictions_above_threshold

    masks, pred_boxes, pred_class = [], [], []
    # process predictions if there are any
    if pred_num > 0:
        masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
        pred_class = [
                INSTANCE_CATEGORY_NAMES[i] for
                i in list(pred[0]['labels'].numpy())
                ]
        pred_boxes = [
                [(i[0], i[1]), (i[2], i[3])] for
                i in list(pred[0]['boxes'].detach().numpy())
                ]
        if len(masks.shape) == 3:
            masks = masks[:pred_num]
        elif len(masks.shape) == 2:
            masks = np.expand_dims(masks, 0)
        pred_boxes = pred_boxes[:pred_num]
        #pred_boxes = [int(coord) for coord_pair in pred_boxes for coord in coord_pair]
        pred_class = pred_class[:pred_num]

    # print the number of detections
    if verbose == 1:
        print("There are {} detected instances.".format(pred_num))

    return masks, pred_boxes, pred_class


def visualize_prediction(img_path: str, masks, boxes, pred_cls,
                         rect_th: float = 3, text_size: float = 3,
                         text_th: float = 3,
                         file_name: str = "inference_result.png"):
    # read image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # create output folder if not present
    create_dir("output/")
    # add bbox and mask to image if present
    if len(masks) > 0:
        for i in range(len(masks)):
            rgb_mask = random_colour_masks(masks[i])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
            cv2.rectangle(img, boxes[i][0], boxes[i][1],
                          color=(0, 255, 0), thickness=rect_th)
            cv2.putText(img, pred_cls[i], boxes[i][0],
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                        thickness=text_th)
    # save inference result
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join("output/", file_name))


def crop_inference_bbox(img_path, masks, boxes,
                        file_name="cropped_inference_result.png"):
    # read image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # create output folder if not present
    create_dir("output/")
    # crop detections
    if len(masks) > 0:
        for i in range(len(masks)):
            cropped_img = img[int(boxes[i][0][1]):int(boxes[i][1][1]),
                              int(boxes[i][0][0]):int(boxes[i][1][0]),
                              :]
            plt.figure(figsize=(20, 30))
            plt.imshow(cropped_img)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join("output/", file_name))


def instance_segmentation_api(image_path: str, weight_path: str):
    # load model
    model = get_model_instance_segmentation(num_classes=2)
    # load model dict
    model_dict = torch.load(weight_path)
    # load weights
    model.load_state_dict(model_dict["state_dict"])
    # load cfg from model dict
    cfg = model_dict["cfg"]
    # get prediction
    masks, boxes, pred_cls = get_prediction(image_path, model, cfg,
                                            threshold=0.75)
    # visualize result
    visualize_prediction(image_path, masks, boxes, pred_cls, rect_th=3,
                         text_size=3, text_th=3)
    # crop detected region
    crop_inference_bbox(image_path, masks, boxes)


if __name__ == '__main__':
    # construct the argument parser
    ap = argparse.ArgumentParser()

    # add the arguments to the parser
    ap.add_argument("image_path", default="test/test_files/CA/CA01_01.tif",
                    help="Path for input image.")
    ap.add_argument("weight_path", default="artifacts/maskrcnn-best.pt",
                    help="Path for trained MaskRCNN model.")
    args = vars(ap.parse_args())

    # perform instance segmentation
    instance_segmentation_api(args['image_path'], args['weight_path'])
