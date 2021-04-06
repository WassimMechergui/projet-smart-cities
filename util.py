from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from math import cos, sin
import random
#----       Utils.py contents     ----#
##### Rotation specific functions #####

def rotate_point(x,y,angle):
    ## Standard usage of angle is in degrees, however math uses radians
    angle = angle * 3.14/180
    return np.array( (x * cos(angle) - y * sin(angle), y * cos(angle) + x * sin(angle) ) )

def rotate_point(point, angle):
    ## Same function but different notations
    angle = angle * 3.14/180
    x , y = point
    return np.array( (  x * cos(angle) - y * sin(angle), y * cos(angle) + x * sin(angle) ) )

def rotate_points(points, angle, offset):
    ## Rotates a list points at once
    offset = np.array(offset)
    return [rotate_point(np.array(point),angle) + offset for point in points]
    
def rotate_image(mat, angle):
    ### Rotates an image (angle in degrees) and expands image to avoid cropping using cv2 warp Affine function

    height, width = mat.shape[:2] # in case image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

    # Extract the offset values, these will be used later on to translate the annotation points
    f_x , f_y = rotation_mat[0,2], rotation_mat[1,2]
    
    return rotated_mat , f_x, f_y

#####----------------------------##### 

#----     Transformations.py     ----#
def extract_bbox(image, mask):
    w,h, _ = mask.shape
    x_min, y_min = 0,0
    x_max, y_max = 0,0
    first = True

    for x in range(w):
        for y in range(h):
            if(mask[x,y,0] != 255 ):
                if(first) :
                    x_max, y_max , x_min, y_min = x,y, x ,y
                    first = False
                else:
                    x_max, y_max = max(x_max,x), max(y_max,y)
                    x_min, y_min = min(x_min,x), min(y_min,y)

    return [x_min, y_min, x_max, y_max]

def center(image,mask):
    w,h, _ = mask.shape
    x_min, y_min = 0,0
    x_max, y_max = 0,0
    first = True

    for x in range(w):
        for y in range(h):
            if(mask[x,y,0] != 255 ):
                if(first) :
                    x_max, y_max , x_min, y_min = x,y, x ,y
                    first = False
                else:
                    x_max, y_max = max(x_max,x), max(y_max,y)
                    x_min, y_min = min(x_min,x), min(y_min,y)
                

    image =  image[x_min:x_max, y_min:y_max, : ] 
    mask  =  mask[x_min:x_max, y_min:y_max, : ]


def extract(image, ann):
    # create a mask with white pixels
    mask = np.ones(image.shape, dtype=np.uint8)
    mask.fill(255)

    for test in ann : 
        area = []
        for i in range(len(test)//2) :
            area.append((test[2*i], test[2*i+1]))
        # points to be cropped
        roi_corners = np.array([area], dtype=np.int32)
        # fill the ROI into the mask
        cv2.fillPoly(mask, roi_corners, 0)

    # applying themask to original image

    masked_image = cv2.bitwise_or(image, mask)

    return masked_image



def extract_ann_image(image, ann_poly):
    ### Extracts the specific spot from an object annotation and centers 

    # Extract image bounding box
    x_min, y_min = int(min([ann_poly[2*i] for i in range(len(ann_poly)//2)])), int(min([ann_poly[2*i+1] for i in range(len(ann_poly)//2)]))
    x_max, y_max = int(max([ann_poly[2*i] for i in range(len(ann_poly)//2)])), int(max([ann_poly[2*i+1] for i in range(len(ann_poly)//2)]))

    y_max = min(image.shape[0] - 1 , y_max)
    x_max = min(image.shape[1] - 1, x_max)

    # Define mask
    mask = np.ones( (y_max-y_min+1 ,x_max-x_min +1,3), dtype=np.uint8)
    mask.fill(255)

    # Extract the image
    extracted_im = image[y_min:y_max+1,x_min:x_max+1,:]

    # Clipping to mask
    area = []    
    nbPoints = len(ann_poly) // 2
    for i in range(len(ann_poly) // 2) :
        area.append((ann_poly[2*i]-x_min, ann_poly[2*i+1]-y_min))
    roi_corners = np.array([area], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, 0)
    

    masked_image = cv2.bitwise_or(extracted_im, mask)

    bbox = [(x_min+x_max)/2, (y_min+y_max)/2, (x_max-x_min)/2, (y_max-y_min)/2]
    poly_couples =  [(ann_poly[2*i]-x_min,ann_poly[2*i+1]-y_min) for i in range(len(ann_poly)//2)]

    return masked_image, mask , poly_couples, bbox 

def insert(image, crop, mask, pos_x, pos_y):
    ### Inserts an object in the format : image to insert into, crop of image to insert, mask, offset position

    crop_h , crop_w = min(crop.shape[0], image.shape[0] - pos_x) , min(crop.shape[1], image.shape[1] - pos_y)
    
    c = image[pos_x: pos_x + crop_h, pos_y : pos_y + crop_w, : ]

    for xx in range(crop_h):
        for yy in range(crop_w):
            if(mask[xx,yy,0] < 1) : c[xx,yy] =   crop[xx,yy]

    result = image[:,:,:]

    result[pos_x: pos_x + crop_h, pos_y : pos_y + crop_w, : ] = c

    return result


def rotate_extracted(crop, mask, key_points , angle):
    ### Rotates an annotation

    ## Rotate mask and extract offset rotation parameters
    rotated_mask  , f_x , f_y= rotate_image( cv2.bitwise_not(mask) , angle)
    rotated_crop , _,_ =  rotate_image( crop , angle)
    ## Resetting Mask to original Black / White configuration
    rotated_mask = cv2.bitwise_not(rotated_mask)
    
    ## Rotating Annotation points and setting offset
    rotated_annotation = rotate_points( key_points, -angle, (f_x,f_y))
    center(rotated_crop,rotated_mask )


    return rotated_crop, rotated_mask, rotated_annotation

def resize_extracted(crop, mask, key_points, r_w, r_h):
    ### Resizes an annotation

    ## Gets height and width
    height, width = crop.shape[:2]

    ## Resizes crop and image
    re_crop = cv2.resize(crop ,(int(r_w*width), int(r_h*height)) , interpolation = cv2.INTER_CUBIC)
    re_mask = cv2.resize(mask ,(int(r_w*width), int(r_h*height)) , interpolation = cv2.INTER_CUBIC)
    
    ## Resizes key_points
    resized_annotation = []

    for point in key_points:
        resized_annotation.append((r_w*point[0],r_h*point[1] ))
        
    return re_crop , re_mask, resized_annotation


def unwrap_points(points, x_off = 0, y_off = 0):
    """ Takes a segment mask polygon in the form [(x,y),(x,y) ... ] and returns it to [x,y,x,y,x,y...]"""
    def offset(i):
        if i%2 == 0 : return x_off
        else : return y_off
    return [points[i//2][i%2] + offset(i) for i in range(2*len(points))]


def generate_random_list(n , max_number = 4):
    if (n == 0):
        return []
    number_of_els = random.randint(1,min(n,max_number))
    pick_list = list(range(n))
    elements = []
    while len(elements) < number_of_els:
        m = random.choice(pick_list)
        pick_list.remove(m)
        elements.append(m)
    return elements