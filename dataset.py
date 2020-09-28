import os
import cv2
import torch
import numpy as np
from utils import process_coco
from albumentations import Compose
from core.coco_utils import convert_coco_poly_to_mask, convert_coco_poly_to_bbox

"""
This dataset loader consumes coco annotation file that includes segmentation masks.
Example coco file format:
    coco_dict = {
            "images": [
                    {"file_name":"data/midv500/images/example1.tif", "id":1},
                    {"file_name":"data/midv500/images/example2.tif", "id":2}
            ],
            "annotations": [
                    {"image_id":1, "category_id":1, "segmentation":[[x1, y1, x2, y2, x3, y3]]},
                    {"image_id":1, "category_id":2, "segmentation":[[x1, y1, x2, y2, x3, y3]]},
                    {"image_id":2, "category_id":1, "segmentation":[[x1, y1, x2, y2, x3, y3]]}
            ],
            "categories": [
                    {'name': 'id_card', 'id': 1},
                    {'name': 'person', 'id': 2}
            ]
    }
"""


class COCODataset(object):
    """
    Compatible with any coco style annotation file, annotations must include
    segmentation mask (polygon coordinates). Bboxes are created from masks.
    Arguments:
        root_dir: Root directory that contains image files. Relative image
        file locations from coco file will be joined with this root_dir while
        iterating.
        coco_path: Path to the coco style annotation file.
        transforms: Albumentations compose object.
    """

    def __init__(self, root_dir: str, coco_path: str, transforms: Compose):
        self.root_dir = root_dir
        self.transforms = transforms
        # process coco file
        images, categories = process_coco(coco_path)
        self.images = images
        self.categories = categories
        self.num_classes = len(self.categories)

    def __getitem__(self, idx):
        # get one image dict from processed coco file
        image_dict = self.images[idx]

        # parse image path
        relative_image_path = image_dict["file_name"]
        # get absolute image path
        abs_image_path = os.path.join(self.root_dir, relative_image_path)
        # load image
        image = cv2.imread(abs_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # parse annotations
        segmentations = []
        category_ids = []

        # find if negative sample
        is_negative_sample = False
        if len(image_dict["annotations"]) == 0:
            is_negative_sample = True
            voc_bboxes = []

        if not is_negative_sample:
            for annotation in image_dict["annotations"]:
                # get segmentation polygons
                segmentations.append(annotation["segmentation"])
                # get category id
                category_id = annotation["category_id"]
                category_ids.append(category_id)

            # create masks from coco segmentation polygons
            masks = convert_coco_poly_to_mask(
                segmentations, height=image.shape[0], width=image.shape[1]
            )

            # create coco and voc bboxes from coco segmentation polygons
            (coco_bboxes, voc_bboxes) = convert_coco_poly_to_bbox(
                segmentations, height=image.shape[0], width=image.shape[1]
            )

            if self.transforms is not None:
                # arrange transform data
                data = {
                    "image": image,
                    "bboxes": voc_bboxes,
                    "masks": masks,
                    "category_id": category_ids,
                }
                # apply transform
                augmented = self.transforms(**data)
                # get augmented image and bboxes
                image = augmented["image"]
                voc_bboxes = augmented["bboxes"]
                category_ids = augmented["category_id"]
                # get masks
                masks = augmented["masks"]

        # check again if augmentation result is negative sample
        if (not is_negative_sample) and (self.transforms is not None):
            if len(augmented["bboxes"]) == 0:
                is_negative_sample = True

        # convert everything into a torch.Tensor
        target = {}

        # boxes
        if (not is_negative_sample) and (
            not voc_bboxes == []
        ):  # if not negative taret and voc_bboxes is not empty
            target["boxes"] = boxes = to_float32_tensor(voc_bboxes)
        else:  # negative target
            target["boxes"] = boxes = torch.zeros((0, 4), dtype=torch.float32)

        # labels
        if not is_negative_sample:  # positive target
            target["labels"] = to_int64_tensor(category_ids)
        else:  # negative target
            target["labels"] = torch.zeros(0, dtype=torch.int64)

        # masks
        if not is_negative_sample:  # positive target
            target["masks"] = to_uint8_tensor(masks)
        else:  # negative target
            target["masks"] = torch.zeros(
                0, image.shape[0], image.shape[1], dtype=torch.uint8
            )

        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["image_id"] = torch.tensor([idx])
        num_objects = len(target["boxes"])
        target["iscrowd"] = torch.zeros((num_objects,), dtype=torch.int64)

        # normalize image
        image = image / np.max(image)
        return image_to_float_tensor(image), target

    def __len__(self):
        return len(self.images)


# class COCODatasetOld(object):
#    """
#    Compatible with any coco style annotation file, annotations must include
#    segmentation mask (polygon coordinates). Bboxes are created from masks.
#    Arguments:
#        root_dir: Root directory that contains image files. Relative image
#        file locations from coco file will be joined with this root_dir while
#        iterating.
#        coco_path: Path to the coco style annotation file.
#        transforms: Albumentations compose object.
#    """
#    def __init__(self, root_dir, coco_path, transforms):
#        self.root_dir = root_dir
#        self.transforms = transforms
#        # process coco file
#        images, categories = process_coco(coco_path)
#        self.images = images
#        self.categories = categories
#        self.num_objects = len(self.categories)
#
#    def __getitem__(self, idx):
#        # get one image dict from processed coco file
#        image_dict = self.images[idx]
#
#        # parse image path
#        relative_image_path = image_dict["file_name"]
#        # form absolute image path
#        abs_image_path = os.path.join(self.root_dir, relative_image_path)
#        # load image
#        image = cv2.imread(abs_image_path)
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#        # parse annotations
#        masks = []
#        voc_bboxes = []
#        category_ids = []
#        for annotation in image_dict["annotations"]:
#            # create mask from poly coords
#            mask_coords = annotation["segmentation"]
#            mask_coords = np.array(mask_coords, dtype=np.int32)
#            mask = np.zeros(image.shape[:2], dtype=np.uint8)
#            cv2.fillPoly(mask, mask_coords.reshape(-1, 4, 2), color=(1))
#            masks.append(mask)
#            # convert coco bbox to voc format for maskrcnn
#            coco_bbox = coco_seg2bbox(annotation["segmentation"])
#            voc_bbox = [coco_bbox[0], coco_bbox[1], coco_bbox[0]+coco_bbox[2], coco_bbox[1]+coco_bbox[3]]
#            voc_bboxes.append(voc_bbox)
#            # get category id
#            category_id = annotation["category_id"]
#            category_ids.append(category_id)
#
#        data = {'image': image, 'bboxes': voc_bboxes, 'masks': masks, 'category_id': category_ids}
#
#        if self.transforms is not None:
#            # apply transform
#            augmented = self.transforms(**data)
#            # get augmented image and bboxes
#            image = augmented["image"]
#            voc_bboxes = augmented["bboxes"]
#            category_ids = augmented["category_id"]
#
#            # convert everything into a torch.Tensor
#            target = {}
#            target["boxes"] = boxes = torch.as_tensor(voc_bboxes, dtype=torch.float32)
#            target["labels"] = torch.as_tensor(category_ids, dtype=torch.int64)
#            target["masks"] = torch.as_tensor(augmented["masks"], dtype=torch.uint8)
#            target["image_id"] = torch.tensor([idx])
#            target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#            target["iscrowd"] = torch.zeros((self.num_objects,), dtype=torch.int64)
#
#        return to_float_tensor(image), target
#
#    def __len__(self):
#        return len(self.images)

# class MIDV500Dataset(object):
#    def __init__(self, root_dir, coco_path, transforms):
#        self.root_dir = root_dir
#        self.transforms = transforms
#        # process coco file
#        images, categories = process_coco(coco_path)
#        self.images = images
#        self.categories = categories
#        self.num_objects = len(self.categories)
#
#    def __getitem__(self, idx):
#        # get one image dict from processed coco file
#        image_dict = self.images[idx]
#
#        # parse fields
#        relative_image_path = image_dict["file_name"]
#        mask_coords = image_dict["annotations"][0]["segmentation"]
#        coco_bbox = image_dict["annotations"][0]["bbox"]
#        voc_bbox = [coco_bbox[0], coco_bbox[1], coco_bbox[0]+coco_bbox[2], coco_bbox[1]+coco_bbox[3]]
#        category_id = image_dict["annotations"][0]["category_id"]
#
#        # form absolute image path
#        abs_image_path = os.path.join(self.root_dir, relative_image_path)
#
#        # load image
#        image = cv2.imread(abs_image_path)
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#        # create mask from poly coords
#        mask_coords = np.array(mask_coords, dtype=np.int32)
#        mask = np.zeros(image.shape[:2], dtype=np.uint8)
#        cv2.fillPoly(mask, mask_coords.reshape(-1, 4, 2), color=(1))
#
#
#        boxes = [voc_bbox]
#        masks = [mask]
#        category_ids = [category_id]
#
#        data = {'image': image, 'bboxes': boxes, 'masks': masks, 'category_id': category_ids}
#
#        if self.transforms is not None:
#            # apply transform
#            augmented = self.transforms(**data)
#            # get augmented image and bboxes
#            image = augmented["image"]
#            bboxes = augmented["bboxes"]
#
#            # convert everything into a torch.Tensor
#            target = {}
#            target["boxes"] = boxes = torch.as_tensor(bboxes, dtype=torch.float32)
#            target["labels"] = torch.ones((self.num_objects,), dtype=torch.int64)
#            target["masks"] = torch.as_tensor(augmented["masks"], dtype=torch.uint8)
#            target["image_id"] = torch.tensor([idx])
#            target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#            target["iscrowd"] = torch.zeros((self.num_objects,), dtype=torch.int64)
#
#        return to_float_tensor(image), target
#
#    def __len__(self):
#        return len(self.images)


def to_float32_tensor(to_be_converted):
    return torch.as_tensor(to_be_converted, dtype=torch.float32)


def to_int64_tensor(to_be_converted):
    return torch.as_tensor(to_be_converted, dtype=torch.int64)


def to_uint8_tensor(to_be_converted):
    return torch.as_tensor(to_be_converted, dtype=torch.uint8)


def image_to_float_tensor(image):
    # Converts numpy images to pytorch format
    return torch.from_numpy(image.transpose(2, 0, 1)).float()
