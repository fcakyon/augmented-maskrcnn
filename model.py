from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import maskrcnn_resnet50_fpn


def get_torchvision_maskrcnn(
    num_classes: int = 91,
    trainable_backbone_layers: int = 3,
    anchor_sizes: list = [32, 64, 128, 256, 512],
    anchor_aspect_ratios: list = [0.5, 1.0, 2.0],
):
    # prepare anchor params
    anchor_sizes = tuple([tuple((anchor_size,)) for anchor_size in anchor_sizes])
    aspect_ratios = tuple(anchor_aspect_ratios)
    aspect_ratios = (aspect_ratios,) * len(anchor_sizes)

    # load an instance segmentation model pre-trained on COCO
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    model = maskrcnn_resnet50_fpn(
        trainable_backbone_layers=trainable_backbone_layers,
        pretrained=True,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_fg_iou_thresh=0.5,
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


# TODO: add custom pretrained model support with configurable trainable layers capability
