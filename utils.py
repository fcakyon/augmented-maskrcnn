import os
import cv2
import json
import yaml
import random
import jsonschema
import numpy as np
from shutil import copyfile

image_schema = {
    "type": "object",
    "properties": {"file_name": {"type": "string"}, "id": {"type": "integer"}},
    "required": ["file_name", "id"],
}

segmentation_schema = {
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}, "additionalItems": False},
    "additionalItems": False,
}

annotation_schema = {
    "type": "object",
    "properties": {
        "image_id": {"type": "integer"},
        "category_id": {"type": "integer"},
        "segmentation": segmentation_schema,
    },
    "required": ["image_id", "category_id", "segmentation"],
}

category_schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "id": {"type": "integer"}},
    "required": ["name", "id"],
}

coco_schema = {
    "type": "object",
    "properties": {
        "images": {"type": "array", "items": image_schema, "additionalItems": False},
        "annotations": {
            "type": "array",
            "items": annotation_schema,
            "additionalItems": False,
        },
        "categories": {
            "type": "array",
            "items": category_schema,
            "additionalItems": False,
        },
    },
    "required": ["images", "annotations", "categories"],
}


def read_and_validate_coco_annotation(coco_annotation_path: str) -> (dict, bool):
    """
    Reads coco formatted annotation file and validates its fields.
    """
    try:
        with open(coco_annotation_path) as json_file:
            coco_dict = json.load(json_file)
        jsonschema.validate(coco_dict, coco_schema)
        response = True
    except jsonschema.exceptions.ValidationError as e:
        print("well-formed but invalid JSON:", e)
        response = False
    except json.decoder.JSONDecodeError as e:
        print("poorly-formed text, not JSON:", e)
        response = False

    return coco_dict, response


def process_coco(coco_file_path: str) -> (list, dict):
    """
    Accepts a coco object detection file.
    Returns list of images and categories.
    """
    # check if coco file is valid and read it
    (coco_dict, response) = read_and_validate_coco_annotation(coco_file_path)

    # raise error if coco file is not valid
    if not (response):
        raise TypeError

    # rearrange coco file for better annotation reach
    images = list()
    for image in coco_dict["images"]:
        image_annotations = list()
        for annotation in coco_dict["annotations"]:
            if image["id"] == annotation["image_id"]:
                image_annotations.append(annotation)
        image["annotations"] = image_annotations
        images.append(image)

    return images, coco_dict["categories"]


def create_dir(_dir):
    """
    Creates given directory if it is not present.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def random_colour_masks(image: np.array):
    """
    Applies random color mask to given input image.
    """
    colours = [
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [80, 70, 180],
        [250, 80, 190],
        [245, 145, 50],
        [70, 150, 250],
        [50, 190, 190],
    ]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    (r[image == 1], g[image == 1], b[image == 1]) = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def visualize_prediction(
    image: str,
    masks,
    boxes,
    pred_cls,
    rect_th: float = 3,
    text_size: float = 3,
    text_th: float = 3,
    file_name: str = "inference_result.png",
):
    """
    Visualizes prediction classes, bounding boxes, masks over the source image
    and exports it to output folder.
    """
    # create output folder if not present
    create_dir("output/")
    # add bbox and mask to image if present
    if len(masks) > 0:
        for i in range(len(masks)):
            rgb_mask = random_colour_masks(masks[i])
            image = cv2.addWeighted(image, 1, rgb_mask, 0.6, 0)
            cv2.rectangle(
                image, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th
            )
            cv2.putText(
                image,
                pred_cls[i],
                boxes[i][0],
                cv2.FONT_HERSHEY_SIMPLEX,
                text_size,
                (0, 255, 0),
                thickness=text_th,
            )
    # save inference result
    save_path = os.path.join("output/", file_name)
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def crop_inference_bbox(image, boxes, file_name="cropped_inference_result"):
    """
    Crops the predicted bounding box regions and exports them to output folder.
    """
    # create output folder if not present
    create_dir("output/")
    # crop detections
    if len(boxes) > 0:
        for ind in range(len(boxes)):
            cropped_img = image[
                int(boxes[ind][0][1]) : int(boxes[ind][1][1]),
                int(boxes[ind][0][0]) : int(boxes[ind][1][0]),
                :,
            ]
            save_path = os.path.join("output/", file_name + "_" + str(ind) + ".png")
            cv2.imwrite(save_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))


def get_category_mapping_from_coco_file(coco_file_path: str) -> dict:
    """
    Creates category id>name mapping from a coco annotation file.
    """
    # check if coco file is valid and read it
    (coco_dict, response) = read_and_validate_coco_annotation(coco_file_path)

    # raise error if coco file is not valid
    if not (response):
        raise TypeError

    coco_categories = coco_dict["categories"]
    category_mapping = {
        str(coco_category["id"]): coco_category["name"]
        for coco_category in coco_categories
    }
    return category_mapping


def split_coco_as_train_val(
    coco_file_path: str, target_dir: str, train_split_rate: float
):
    """
    Takes single coco dataset file path, split images into train-val and saves as seperate coco dataset files.
    """
    # check if coco file is valid and read it
    (coco_dict, response) = read_and_validate_coco_annotation(coco_file_path)

    # raise error if coco file is not valid
    if not (response):
        raise TypeError

    # divide coco dict into train val coco dicts
    num_images = len(coco_dict["images"])
    random_indices = np.random.permutation(num_images).tolist()
    num_train = int(num_images * train_split_rate)
    # divide images
    train_indices = random_indices[:num_train]
    val_indices = random_indices[num_train:]
    train_images = np.array(coco_dict["images"])[
        (np.array(train_indices) - 1).tolist()
    ].tolist()
    val_images = np.array(coco_dict["images"])[
        (np.array(val_indices) - 1).tolist()
    ].tolist()
    # divide annotations
    train_annotations = list()
    val_annotations = list()
    for annotation in coco_dict["annotations"]:
        if annotation["image_id"] in train_indices:
            train_annotations.append(annotation)
        elif annotation["image_id"] in val_indices:
            val_annotations.append(annotation)
    # form train val coco dicts
    train_coco_dict = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_dict["categories"],
    }
    val_coco_dict = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco_dict["categories"],
    }
    # get filename of the base coco file
    base_coco_filename = os.path.basename(coco_file_path).replace(".json", "")
    # save train val coco files
    save_json(
        train_coco_dict, os.path.join(target_dir, base_coco_filename + "_train.json")
    )
    save_json(val_coco_dict, os.path.join(target_dir, base_coco_filename + "_val.json"))


def save_json(data, save_path):
    """
    Saves json formatted data (given as "data") as save_path
    Example inputs:
        data: {"image_id": 5}
        save_path: "dirname/coco.json"
    """
    # type check when save json files
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(NumpyEncoder, self).default(obj)

    # create dir if not present
    save_dir = os.path.dirname(save_path)
    create_dir(save_dir)

    # export as json
    with open(save_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=4, separators=(",", ": "), cls=NumpyEncoder)


def load_json(load_path):
    """
    Loads json formatted data (given as "data") from load_path
    Example inputs:
        load_path: "dirname/coco.json"
    """
    # read from path
    with open(load_path) as json_file:
        data = json.load(json_file)
    return data


def read_yaml(yaml_path):
    """
    Reads yaml file as dict.
    """
    with open(yaml_path) as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    return yaml_data


def save_yaml(dict_file, yaml_path):
    """
    Saves a dict as yaml file.
    """
    with open(yaml_path, "w") as file:
        documents = yaml.dump(dict_file, file)


class Configuration:
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    default_config_path = os.path.join(
        current_file_path, "configs", "default_config.yml"
    )

    def __init__(
        self, config_path: str = None, default_config_path=default_config_path
    ):
        base_config = read_yaml(default_config_path)  # read base config
        if config_path is not None:
            config = read_yaml(config_path)
            base_config.update(config)  # overwrite base config
        self.as_dict = base_config  # set overwritten config
