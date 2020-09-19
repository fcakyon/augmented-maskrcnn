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


def get_transforms(config, mode: str = "predict") -> Compose:
    """
    Composes albumentations transforms.
    Returns the full list of transforms when mode is "train".
    mode should be one of "train", "val" or "predict".
    """
    # compose prediction transforms
    if mode == "predict":
        transforms = Compose(
            [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
        )
    # compose validation transforms
    elif mode == "val":
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
    # TODO: make transformation parameters configurable from yml
    elif mode == "train":
        transforms = Compose(
            [
                LongestMaxSize(
                    max_size=config["LONGESTMAXSIZE_MAXSIZE"],
                    p=config["LONGESTMAXSIZE_P"],
                ),
                # PadIfNeeded(min_height=768, min_width=768, border_mode=0, p=1),
                RandomSizedBBoxSafeCrop(
                    height=config["RANDOMSIZEDBBOXSAFECROP_HEIGHT"],
                    width=config["RANDOMSIZEDBBOXSAFECROP_WIDTH"],
                    p=config["LONGESTMAXSIZE_P"],
                ),
                HorizontalFlip(p=config["HORIZONTALFLIP_P"]),
                RandomRotate90(p=config["RANDOMROTATE90_P"]),
                RandomBrightnessContrast(
                    brightness_limit=config["RANDOMBRIGHTNESSCONTRAST_BRIGHTNESSLIMIT"],
                    contrast_limit=config["RANDOMBRIGHTNESSCONTRAST_CONTRASTLIMIT"],
                    p=config["RANDOMBRIGHTNESSCONTRAST_P"],
                ),
                RandomGamma(
                    gamma_limit=config["RANDOMGAMMA_GAMMALIMIT"],
                    p=config["RANDOMGAMMA_P"],
                ),
                HueSaturationValue(
                    hue_shift_limit=config["HUESATURATIONVALUE_HUESHIFTLIMIT"],
                    sat_shift_limit=config["HUESATURATIONVALUE_SATSHIFTLIMIT"],
                    val_shift_limit=config["HUESATURATIONVALUE_VALSHIFTLIMIT"],
                    p=config["HUESATURATIONVALUE_P"],
                ),
                MotionBlur(
                    blur_limit=tuple(config["MOTIONBLUR_BLURLIMIT"]),
                    p=config["MOTIONBLUR_P"],
                ),
                JpegCompression(
                    quality_lower=config["JPEGCOMPRESSION_QUALITYLOWER"],
                    quality_upper=config["JPEGCOMPRESSION_QUALITYUPPER"],
                    p=config["JPEGCOMPRESSION_P"],
                ),
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
