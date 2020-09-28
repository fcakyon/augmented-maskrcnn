import unittest


class Tests(unittest.TestCase):
    def test_read_and_validate_coco_annotation(self):
        from utils import read_and_validate_coco_annotation

        false_sample_list = [
            "tests/data/coco_false_" + str(ind) + ".json" for ind in range(17)
        ]
        true_sample_list = [
            "tests/data/coco_true_" + str(ind) + ".json" for ind in range(2)
        ]

        for false_sample in false_sample_list:
            _, response = read_and_validate_coco_annotation(false_sample)
            self.assertEqual(response, False)

        for true_sample in true_sample_list:
            _, response = read_and_validate_coco_annotation(true_sample)
            self.assertEqual(response, True)

    def test_process_coco(self):
        import jsonschema
        from utils import process_coco

        # form json schemas
        segmentation_schema = {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
                "additionalItems": False,
            },
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

        image_schema = {
            "type": "object",
            "properties": {
                "file_name": {"type": "string"},
                "id": {"type": "integer"},
                "annotations": {
                    "type": "array",
                    "items": annotation_schema,
                    "additionalItems": False,
                },
            },
            "required": ["file_name", "id"],
        }

        image_list_schema = {
            "type": "array",
            "items": image_schema,
            "additionalItems": False,
        }

        category_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "id": {"type": "integer"}},
            "required": ["name", "id"],
        }

        category_list_schema = {
            "type": "array",
            "items": category_schema,
            "additionalItems": False,
        }

        # process sample coco file
        COCO_PATH = "tests/data/coco_true_0.json"
        images, categories = process_coco(COCO_PATH)

        # check if returned list lenghts are valid
        self.assertEqual(len(images), 2)
        self.assertEqual(len(categories), 2)

        # check if returned image fileds are valid
        self.assertEqual(images[1]["id"], 2)
        self.assertEqual(images[1]["file_name"], "data/midv500/images/example2.tif")
        self.assertEqual(images[1]["annotations"][0]["image_id"], 2)
        self.assertEqual(images[0]["annotations"][1]["image_id"], 1)
        self.assertEqual(images[0]["annotations"][1]["category_id"], 2)

        # check if returned images schema is valid
        try:
            jsonschema.validate(images, image_list_schema)
            validation = True
        except jsonschema.exceptions.ValidationError as e:
            print("well-formed but invalid JSON:", e)
            validation = False
        self.assertEqual(validation, True)

        # check if returned categories schema is valid
        try:
            jsonschema.validate(categories, category_list_schema)
            validation = True
        except jsonschema.exceptions.ValidationError as e:
            print("well-formed but invalid JSON:", e)
            validation = False
        self.assertEqual(validation, True)

    def test_COCODataset(self):
        from transform import get_transforms
        from dataset import COCODataset
        from utils import read_yaml

        # read config file for transforms
        config_path = "configs/default_config.yml"
        config = read_yaml(config_path)
        # form basic albumentation transform
        transforms = get_transforms(config=config, mode="val")

        # init COCODataset
        DATA_ROOT = "tests/data/"
        COCO_PATH = "tests/data/coco_true_1.json"
        dataset = COCODataset(DATA_ROOT, COCO_PATH, transforms)

        # iterate over the dataset
        # and get annotations (target dict) for the first image
        image_tensor = next(iter(dataset))[0]
        target_tensor = next(iter(dataset))[1]

        # apply checks for image tensor
        self.assertEqual(image_tensor.type(), "torch.FloatTensor")
        self.assertEqual(list(image_tensor.size()), [3, 1920, 1080])
        self.assertAlmostEqual(float(image_tensor.max()), 1.0, places=2)
        self.assertAlmostEqual(float(image_tensor.mean()), 0.39, places=2)

        # apply checks for each field in the target tensor dict
        boxes_tensor_0 = target_tensor["boxes"][0]
        self.assertEqual(boxes_tensor_0.type(), "torch.FloatTensor")
        self.assertEqual(
            boxes_tensor_0.cpu().numpy().tolist(), [97.0, 643.0, 931.0, 1185.0]
        )

        labels_tensor_0 = target_tensor["labels"][0]
        self.assertEqual(labels_tensor_0.type(), "torch.LongTensor")
        self.assertEqual(labels_tensor_0.cpu().numpy().item(), 1)

        masks_tensor_0 = target_tensor["masks"][0]
        self.assertEqual(masks_tensor_0.type(), "torch.ByteTensor")
        self.assertEqual(list(masks_tensor_0.size()), [1920, 1080])
        self.assertAlmostEqual(float(masks_tensor_0.max()), 1.0, places=1)

        image_id_tensor_0 = target_tensor["image_id"][0]
        self.assertEqual(image_id_tensor_0.type(), "torch.LongTensor")
        self.assertEqual(image_id_tensor_0.cpu().numpy().item(), 0)

        area_tensor_0 = target_tensor["area"][0]
        self.assertEqual(area_tensor_0.type(), "torch.FloatTensor")
        self.assertEqual(area_tensor_0.cpu().numpy().item(), 452028.0)

        iscrowd_tensor_0 = target_tensor["iscrowd"][0]
        self.assertEqual(iscrowd_tensor_0.type(), "torch.LongTensor")
        self.assertEqual(iscrowd_tensor_0.cpu().numpy().item(), 0)

        boxes_tensor_1 = target_tensor["boxes"][1]
        self.assertEqual(boxes_tensor_1.type(), "torch.FloatTensor")
        self.assertEqual(
            boxes_tensor_1.cpu().numpy().tolist(), [97.0, 500.0, 931.0, 1185.0]
        )

        labels_tensor_1 = target_tensor["labels"][1]
        self.assertEqual(labels_tensor_1.type(), "torch.LongTensor")
        self.assertEqual(labels_tensor_1.cpu().numpy().item(), 2)

        masks_tensor_1 = target_tensor["masks"][1]
        self.assertEqual(masks_tensor_1.type(), "torch.ByteTensor")
        self.assertEqual(list(masks_tensor_1.size()), [1920, 1080])
        self.assertAlmostEqual(float(masks_tensor_1.max()), 1.0, places=1)

        area_tensor_1 = target_tensor["area"][1]
        self.assertEqual(area_tensor_1.type(), "torch.FloatTensor")
        self.assertEqual(area_tensor_1.cpu().numpy().item(), 571290.0)


if __name__ == "__main__":
    unittest.main()
