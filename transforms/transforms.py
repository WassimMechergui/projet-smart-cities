# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
import sys

## Copy Paste specific imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import util
import random
import time
##


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target,image_to_copy= None , target_to_copy = None):
        for t in self.transforms:
            #try : 
                
                if(hasattr(t, "name")):
                    
                    image, target = t(image, target, image_to_copy, target_to_copy)
                else:
                    image, target = t(image, target)
            #except:
             #   print("Unexpected error:", sys.exc_info()[0])
              #  image, target = t(image, target)
        
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target



class RandomCopyPaste(object):
    def __init__(self, min_width = 10 , min_height = 10, activated = True, transforms = []):
        self.name = "Random Copy Paste"
        self.minimum_width = min_width
        self.minimum_height = min_height
        self.activated = activated
        self._transforms= transforms

    def __call__(self,img, anno, img_copy = None, anno_copy = None):
        if(self.activated == False):
            
            return img, anno

        img, img_copy = np.array(img).copy() , np.array(img_copy)
        ## STEP 0 : pickout which indexes to copy

        list_copy_idx = util.generate_random_list(len(anno_copy), max_number = 4)

        ## STEP 1 : extract segments ,classes, keypoints
        segments , key_points,indices  = [] , [], []
        
        for idx in list_copy_idx:
            try:
                segment =  anno_copy[idx]['segmentation'][0]
                if('keypoints' in anno_copy[idx]):
                    key_point = anno_copy[idx]['keypoints'][0]
                else:
                    key_point = []
                
                segments.append(segment)
                key_points.append(key_point)
                indices.append(idx)
            except: 
                pass

        ## STEP 2 : extract specific crops & masks and insert them into the original image


        new_annotations = []
        for idx in range(len(segments)):
            ann_poly = segments[idx]

            ## STEP i : extract the specific crop and mask and bbox
            cropped, mask, poly_points, bbox  = util.extract_ann_image(img_copy, ann_poly)
            if(cropped.shape[0] < self.minimum_height or cropped.shape[1] < self.minimum_width) or (cropped.shape[0] > 200 and cropped.shape[1] > 200  ):
                continue
            else:

                ## STEP iii : apply transforms on cropped out annotations
                for t in self._transforms:
                    cropped, mask, poly_points = t(cropped, mask, poly_points)


                ## STEP iii : insert into image

                pos_x ,pos_y = random.randint( 0, max(img.shape[1] - cropped.shape[1]//5-1,0) ),random.randint(0, max(img.shape[0] - cropped.shape[0]//5-1,0)  )
                util.insert(img, cropped , mask , pos_x = pos_x, pos_y = pos_y) 
                bbox = util.extract_bbox(cropped, mask)
                bbox = [bbox[0] +pos_x, bbox[1] + pos_y, bbox[2] + pos_x, bbox[3] + pos_y]
                

                ## STEP iv: insert into annotation
                new_annotation = {'area': anno_copy[indices[idx]]['area'] ,
                                'bbox' : bbox,
                                'category_id': anno_copy[indices[idx]]['category_id'],
                                'segmentation' : [util.unwrap_points(poly_points, pos_x, pos_y)],
                                'iscrowd' : anno_copy[indices[idx]]['iscrowd']
                                }
                if(key_points != []): new_annotation.update({'keypoints': key_points[idx]})
                
                new_annotations.append(new_annotation)
        ## STEP 3: add anotations and convert image back to PIL format
        anno = anno+ new_annotations

        img = Image.fromarray(img)

        return img, anno
    

class PolygonSpecific_rotate(object):
    def __init__(self, min_angle = -30, max_angle = 30, probability= 0.5):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.p = probability
    def __call__(self, crop, mask, polygon_points):
        if(random.random() < self.p):
            return util.rotate_extracted(crop, mask, polygon_points , float(random.randint(self.min_angle, self.max_angle)))
        else:
            return crop ,mask, polygon_points


class PolygonSpecific_resize(object):
    def __init__(self, min_ratio = 0.5, max_ratio = 2.0, probablity = 0.5):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.p = probablity
    def __call__(self, crop, mask, polygon_points):
        if(random.random() < self.p):
            ratio =  random.uniform(self.min_ratio, self.max_ratio)
            return util.resize_extracted(crop, mask, polygon_points , ratio, ratio)
        else:
            return crop ,mask, polygon_points