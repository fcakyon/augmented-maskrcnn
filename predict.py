import cv2
import torch
import argparse
import numpy as np
from albumentations import (
    Compose,
    Normalize
)
from utils import (visualize_prediction, crop_inference_bbox)
from train import get_model_instance_segmentation


def get_transform() -> Compose:
    transforms = Compose([Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    return transforms


def to_float_tensor(img: np.array) -> torch.tensor:
    # Converts numpy images to pytorch format
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


def get_prediction(image: np.array, model, category_mapping: dict = {},
                   threshold: float = 0.5,
                   verbose: int = 1) -> (list, list, list):
    # apply transform
    transforms = get_transform()
    augmented = transforms(image=image)
    image = augmented["image"]
    # convert to tensor
    image = to_float_tensor(image).unsqueeze(0)

    # get prediction
    model.eval()
    pred = model(image)

    # map prediction ids to labels if category_mapping is given as input
    if not(category_mapping == {}):
        INSTANCE_CATEGORY_NAMES = category_mapping
    else:
        INSTANCE_CATEGORY_NAMES = {ind: ind for ind in range(999)}

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
                [(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for
                i in list(pred[0]['boxes'].detach().numpy())
                ]
        if len(masks.shape) == 3:
            masks = masks[:pred_num]
        elif len(masks.shape) == 2:
            masks = np.expand_dims(masks, 0)
        pred_boxes = pred_boxes[:pred_num]
        pred_class = pred_class[:pred_num]

    # print the number of detections
    if verbose == 1:
        print("There are {} detected instances.".format(pred_num))

    return masks, pred_boxes, pred_class


def instance_segmentation_api(image_path: str, weight_path: str):
    # read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # load model dict
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dict = torch.load(weight_path, map_location=DEVICE)
    # load cfg from model dict
    cfg = model_dict["cfg"]
    # load model
    model = get_model_instance_segmentation(num_classes=cfg["NUM_CLASSES"])
    # load weights
    model.load_state_dict(model_dict["state_dict"])

    # get prediction
    masks, boxes, pred_cls = get_prediction(
            image,
            model,
            category_mapping=cfg["CATEGORY_MAPPING"],
            threshold=0.75)

    # visualize result
    visualize_prediction(image, masks, boxes, pred_cls, rect_th=3,
                         text_size=3, text_th=3)
    # crop detected region
    crop_inference_bbox(image, boxes)


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
