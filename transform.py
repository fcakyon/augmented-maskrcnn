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


def get_transforms(train: bool) -> Compose:
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
