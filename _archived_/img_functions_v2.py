import tensorflow as tf

#--- ----- Image Displaying

# display image/s using matplotlib.pyplot
# https://www.tensorflow.org/tutorials/images/segmentation

from matplotlib import pyplot as plt

def display_images(display_list:list):
    plt.figure(figsize=(15, 15))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        #plt.axis('off')
    plt.show()

def print_image(image: list):
    for row_of_pixels in image:
        print(row_of_pixels)

#--- ----- Image Processing

### create list of all images inside a folder path + its subfolders

import os
import cv2

def get_all_image_names(file_path:str):
    image_names = []
    
    for roots,dirs,files in os.walk(file_path):
        for fn in files:
            # check each filename if it falls under any of the valid datatypes
            if fn.endswith('.jpg') or fn.endswith('.png'):
                image_path = os.path.join(roots,fn)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images_rgb.append(image)

    return images_rgb

def get_all_images_rgb(file_path:str):
    images_rgb = []
    
    for roots,dirs,files in os.walk(file_path):
        for fn in files:
            # check each filename if it falls under any of the valid datatypes
            if fn.endswith('.jpg') or fn.endswith('.png'):
                image_path = os.path.join(roots,fn)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images_rgb.append(image)

    return images_rgb

#---

### convert True Mask to Land Use (numerical)

def pixel_rgb_to_index(rgb_list:list, lookup_table:dict):
    # converts a 3x1 list [R, G, B] into a 3x1 tuple (R, G, B)
    # the tuple is then processed in the lookup dictionary
    #       to get a single number for the actual classification
    
    return lookup_table.get(tuple(rgb_list), len(lookup_table))

def image_rgb_to_index(rgb_image:list, lookup_table:dict):
    # converts a HEIGHTxWIDTHx3 (in this case 20x20) tensor into
    #       a HEIGHTxWIDTH 2D list
    index_image = []

    for row_of_pixels in rgb_image:
        current_row = []
        
        for pixel in row_of_pixels:
            current_row.append(pixel_rgb_to_index(pixel, lookup_table))

        index_image.append(current_row)            

    return index_image

#---

### convert image into grid of subdivided images

small_sample = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]

large_sample = []
for i in range(1000):
    large_sample.append([i for i in range(1000)])
    
#---

### compile images and masks to fit with CNN

def load_image_train(datapoint:tuple):
    # datapoint is just some iterable with both an image and a segmentation mask
    # examples from tensorflow use iterable classes for these
    # datapoint is usually a tuple as (image, mask)
    
    src_image = datapoint['image']
    src_mask = datapoint['segmentation_mask']
    input_image = tf.image.resize(src_image, (img_HEIGHT, img_WIDTH))
    input_mask = tf.image.resize(src_mask, (img_HEIGHT, img_WIDTH))

    #augment_data(input_image, input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(datapoint:tuple):
    # see above re: datapoints
    
    src_image = datapoint['image']
    src_mask = datapoint['segmentation_mask']
    input_image = tf.image.resize(src_image, (img_HEIGHT, img_WIDTH))
    input_mask = tf.image.resize(src_mask, (img_HEIGHT, img_WIDTH))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def normalize(image:list, mask:list):
    # divides the RGB values of the image by 255
    # changes the range of values in the tensor from 0-255 into 0-1
    image = tf.cast(image, tf.float32) / 255.0

    return image, mask


##define dataAugmentation:
##    let x be a random number from [0,1]
##    if > 0.5:
##        flip the image
##    else:
##        rotate the image by a random degree
##

def augment_data(image:list, mask:list):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(image)
        input_mask = tf.image.flip_left_right(mask)
        pass

    # todo : clarify if do all 4 rotations from 0 to 270?
    # still needs fixing with regards to actually increasing the number of images
