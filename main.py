import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import os
import cv2

import PIL
from matplotlib import pyplot as plt

#--- -----

# display image/s using matplotlib.pyplot
# https://www.tensorflow.org/tutorials/images/segmentation

def display(display_list):
    plt.figure(figsize=(15, 15))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        #plt.axis('off')
    plt.show()

#--- ----- Prerequisite functions for the CNN
    
### create list of all images inside a folder URL + its subfolders
    
def get_all_images(file_url):
    images = []
    
    for roots,dirs,files in os.walk(file_url):
        for fn in files:
            # check each filename if it falls under any of the valid datatypes
            if fn.endswith('.jpg') or fn.endswith('.png'):
                image_url = os.path.join(roots,fn)
                image = cv2.imread(image_url)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)

    return images

#---

### True Mask to Land Use conversion (numerical)

lookup_color_to_label = {
    (97,64,31): 0,	    # agricultural - brown
    (160,32,239): 1,	    # commercial - purple
    (221,190,170): 2,       # industrial - beige
    (237,0,0): 3,   	    # institutional - red
    (45,137,86): 4,	    # recreational - green
    (254,165,0): 5,	    # residential - yellow
    (0,0,87): 6		    # transport - dark blue

    # (0,0,0): 0 = none
}

def pixel_rgb_to_index(rgb_list):
    # converts a 3x1 list [R, G, B] into a 3x1 tuple (R, G, B)
    # the tuple is then processed in the lookup dictionary
    #       to get a single number for the actual classification
    
    return lookup_color_to_label.get(tuple(rgb_list), len(lookup_color_to_label))

def image_rgb_to_index(rgb_image):
    # for each pixel in image, convert?
    # a nested loop approach may be terribly slow

    # todo: maybe find a more efficient way to convert [480x480x3] tensor
    #       from an RGB image, into a [480x480] with just the classifications
    pass

#---

### Image Processing for making CNN Datasets

def load_image_train(datapoint):
    # datapoint is just some iterable with both an image an a segmentation mask
    # examples from tensorflow use iterable classes for these
    # the later functions will instantiate tuples for the datapoints
    
    src_image = datapoint['image']
    src_mask = datapoint['segmentation_mask']
    input_image = tf.image.resize(src_image, (img_HEIGHT, img_WIDTH))
    input_mask = tf.image.resize(src_mask, (img_HEIGHT, img_WIDTH))

    #augment_data(input_image, input_mask)

    #input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(datapoint):
    # see above re: datapoints
    
    src_image = datapoint['image']
    src_mask = datapoint['segmentation_mask']
    input_image = tf.image.resize(src_image, (img_HEIGHT, img_WIDTH))
    input_mask = tf.image.resize(src_mask, (img_HEIGHT, img_WIDTH))

    #input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def normalize(image, mask):
    # divides the RGB values of the image by 255
    # changes the range of values in the tensor from 0-255 into 0-1
    image = tf.case(image, tf.float32) / 255.0

    return image, mask


##define dataAugmentation:
##    let x be a random number from [0,1]
##    if > 0.5:
##        flip the image
##    else:
##        rotate the image by a random degree
##

def augment_data(image, mask):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(image)
        input_mask = tf.image.flip_left_right(mask)
        pass

    # todo : clarify if do all 4 rotations from 0 to 270?
    # still needs fixing with regards to actually increasing the number of images

#--- ----- The actual CNN

img_HEIGHT = 480
img_WIDTH = 480

TRAIN_SIZE = 2500
FOLDS = 5
TEST_SIZE = 500

BATCH_SIZE = 64                             # power of 2
ITERATIONS = -(-TRAIN_SIZE // BATCH_SIZE)   # ceiling division
EPOCHS = 727                                # filler number, just has to be more than enough to overfit before reaching the final epoch

#---

### Training + Validation Sets

train_images_DIR = os.path.join('train_imgdata', 'satellite_4800x4800')
train_truemasks_DIR = os.path.join('train_imgdata', 'truemask_480x480')

train_images = get_all_images(train_images_DIR)
train_truemasks_color = get_all_images(train_truemasks_DIR)
train_truemasks_index = []

train_samples = list(zip(train_images, train_images))
    # (image, mask) tuple x 2500

    # pray that get_all_files() preserves the image order when applied
    #   between train_images and train_truemasks independently

    # todo: remember to set to (train_images, train_truemasks_index) eventually


# see above, image_rgb_to_index()
##for row in train_truemasks_color[0]:
##    for column in row:
##        print(column)
##        #column = lookup_color_to_label.get(column, 7)

#---

### Test Set

test_images_DIR = os.path.join('test_imgdata', 'satellite_480x480')
test_truemasks_DIR = os.path.join('test_imgdata', 'truemask_480x480')

test_images = get_all_images(test_images_DIR)
test_truemasks_color = get_all_images(test_truemasks_DIR)
test_truemasks_index = []

test_samples = list(zip(test_images, test_images))
    # (image, mask) tuple x 500

    # todo: remember to set to (test_images, test_truemasks_index) eventually

#---

### sample segmentation dataset from tensorflow

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
s = train.take(1)

for sample in s:
    print(sample[1].numpy())
    print(sample[1][240][240])      # pet = 1
    print(sample[1][0][0])          # background = 2
    print(sample[1][200][100])      # border = 3

    display([sample[0], sample[1]])
    pass

#---

def buildCNN():
    pass

##define buildCNN:
##    //relu will be used as the activation function for all the layers
##    //x = max(0, x)
##    create conv layer 1_1
##    create conv layer 1_2
##    max pool output of conv layer 1_2
##    create conv layer 2_1
##    create conv layer 2_2
##    max pool output of conv layer 2_2
##        ...
##    create conv layer n_1
##    create conv layer n_2
##    max pool output of conv layer n_2
##    //add layers until overfitting occurs
##    create conv layer n+1_1
##    create conv layer n+1_2
##    upsample the output of conv layer n+1_2
##        ...
##    create conv layer 2n_1
##    create conv layer 2n_2
##    upsample the output of conv layer 2n_2
##
##    flatten the output
##    create a dense layer
##    //last layer
##    convert the tensor into an image
##    display input image, true mask, and predicted mask
##

#--- ----- leftover pseudocode

##import images and color maps as tf dataset
##
##//color maps would have pixels that are labeled from 0-6
##divide the rgb values of the images by 255
##
##//used to ensure that the rgb values are within [0,1]
##
##divide the dataset into training dataset and testing dataset
##use data augmentation on the training dataset
##
##divide the training dataset into five for k-fold cross validation
##for validationSet in trainingDataset:
##    for e from 1 to epoch:
##    build CNN using buildCNN
##train the model using the four training datasets, leaving the validationSet to act as a validation set
##
##    //loss function will be sparse categorical cross entropy
##    evaluate the model using the validation set
##    record evaluation
