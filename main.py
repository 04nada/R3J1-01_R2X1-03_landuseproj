import img_functions as img_funcs

import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import PIL

import os
import cv2

#--- ----- CNN Parameters

### image parameters

img_COLORMAP_HEIGHT = 20
img_COLORMAP_WIDTH = 20

img_HEIGHT = 480
img_WIDTH = 480

lookup_rgb_to_index = {
    (97,64,31): 0,	    # agricultural - brown - #61401F
    (160,32,239): 1,	    # commercial - purple - #A020EF
    (221,190,170): 2,       # industrial - beige - #DDBEAA
    (237,0,0): 3,   	    # institutional - red - #ED0000
    (45,137,86): 4,	    # recreational - green - #2D8956
    (254,165,0): 5,	    # residential - yellow - #FEA500
    (0,0,87): 6		    # transport - dark blue - #000057
}

#---

### model interation parameters

SEED = 727                                      # consistent randomization from a set seed

TRAIN_SIZE = 2500                               
FOLDS = 5                                       
BATCH_SIZE = 32                                 # power of 2 for optimized CPU/GPU usage
STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE      # floor division

TEST_SIZE = 500                                 

EPOCHS = 1000                                   # filler number, just has to be more than enough to overfit before reaching the final epoch

#---

### model implementation parameters

ACTIVATION = 'relu'

OPTIMIZER = 'sgd'                                           # Stochastic Gradient Descent
LOSS = tf.keras.losses.SparseCategoricalCrossentropy()      # Sparce Categorical Cross-Entropy
EVALUATION_METRICS = [
    tf.keras.metrics.MeanIoU()
]

#--- ----- CNN Implementation

tf.random.set_seed(SEED)


### Training + Validation Sets

train_images_DIR = os.path.join('train_imgdata', 'satellite_4800x4800_sortbyname')
train_truemasks_DIR = os.path.join('train_imgdata', 'trueclass_240x240_sortbyclass')

train_images = img_funcs.get_all_images_rgb(train_images_DIR)
train_truemasks_color = img_funcs.get_all_images_rgb(train_truemasks_DIR)
train_truemasks_index = []

train_dataset = list(zip(train_images, train_images))
    # (image, mask) tuple x 2500

    # pray that get_all_files() preserves the image order when applied
    #   between train_images and train_truemasks independently

    # todo: remember to set to (train_images, train_truemasks_index) eventually

kf = KFold(n_splits=FOLDS, shuffle=True)
#train_dataset_folds = list(kf.split(train_dataset))

list_100 = [i for i in range(100)]
list_100_folds = list(kf.split(list_100))


### CNN Training

for f in range(FOLDS):
    current_trainset = list_100_folds[f][0]
    current_valset = list_100_folds[f][1]

    print(current_valset)

    # only do training augmentation here, so it will happen FOLDSx total
    # in order to augment training data but not the validation data, per fold
    # https://stats.stackexchange.com/questions/482787/how-to-do-data-augmentation-and-cross-validation-at-the-same-time

##    model = Sequential()
##    
##    model.add(idk
    # todo: model.add() a lot of shtuff
    
##    model.fit(
##        current_trainset
##        batch_size = BATCH_SIZE
##        epochs=1
##        validation_data = current_valset
##            
##    model.compile(
##        optimizer=OPTIMIZER,
##        loss=LOSS,
##        metrics=METRICS
##    )

    # todo: figure out how to combine results per fold
#---

### Test Set

test_images_DIR = os.path.join('test_imgdata', 'satellite_480x480')
test_truemasks_DIR = os.path.join('test_imgdata', 'truemask_480x480')

test_images = img_funcs.get_all_images_rgb(test_images_DIR)
test_truemasks_color = img_funcs.get_all_images_rgb(test_truemasks_DIR)
test_truemasks_index = [image_rgb_to_index(img) for img in test_truemasks_color]

test_dataset = list(zip(test_images, test_images))
    # (image, mask) tuple x 500

    # same prayers as above with the Training/Validation Set
    # todo: remember to set to (test_images, test_truemasks_index) eventually

#---

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

# ---

##base_model = tf.keras.applications.MobileNetV2(
##    input_shape=[128, 128, 3],
##    include_top=False
##)
##
### Use the activations of these layers
##layer_names = [
##    'block_1_expand_relu',   # 64x64
##    'block_3_expand_relu',   # 32x32
##    'block_6_expand_relu',   # 16x16
##    'block_13_expand_relu',  # 8x8
##    'block_16_project',      # 4x4
##]
##base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
##
### Create the feature extraction model
##down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
##
##down_stack.trainable = False

#--- ----- drafted pseudocode

    #+# import images and color maps as tf dataset

    #+# //color maps would have pixels that are labeled from 0-6
    #+# divide the rgb values of the images by 255
    #+# 
    #+# //used to ensure that the rgb values are within [0,1]

    #+# divide the dataset into training dataset and testing dataset

    ##use data augmentation on the training dataset

    #+# divide the training dataset into five for k-fold cross validation

    #+# for validationSet in trainingDataset:
    #+#     for e from 1 to epoch:
    ##          build CNN using buildCNN
    ##          train the model using the four training datasets, leaving the validationSet to act as a validation set

    #+# for f from 0 to 4:
    #+#     current_valid = folds[f]
    #+#     current_train = everything else

##    //loss function will be sparse categorical cross entropy
##    evaluate the model using the validation set
##    record evaluation
