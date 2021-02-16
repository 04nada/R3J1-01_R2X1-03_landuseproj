import tensorflow as tf
import numpy as np

img_HEIGHT = 480
img_WIDTH = 480

train_satlt_dir = './train_imgdata/satellite_480x480'
train_truem_dir = './train_imgdata/truemask_480x480'
FOLDS = 5

test_satlt_dir = './test_imgdata/satellite_480x480'
test_truem_dir = './test_imgdata/truemask_480x480'


def buildCNN:
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
##define dataAugmentation:
##    let x be a random number from [0,1]
##    if > 0.5:
##        flip the image
##    else:
##        rotate the image by a random degree
##
##import images and color maps as tf dataset
##
##//color maps would have pixels that are labeled from 0-6
##divide the rgb values of the images by 255
##
##//used to ensure that the rgb values are within [0,1]
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
