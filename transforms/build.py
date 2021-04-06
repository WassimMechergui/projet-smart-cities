# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
       pass
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0



    transform = T.Compose(
        [

            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),

        ]
    )
    return transform


def build_copy_paste(is_train= True):

    rotate = T.PolygonSpecific_rotate(min_angle = -80, max_angle = 80, probability= 0.5)
    resize = T.PolygonSpecific_resize(min_ratio = 0.5, max_ratio = 2.0, probablity = 0.5)
    transforms = [rotate, resize]

    copy_paste = T.RandomCopyPaste(activated = is_train, transforms = transforms)

    return copy_paste


