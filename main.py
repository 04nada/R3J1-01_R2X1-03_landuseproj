import main_params as mp
import img_functions as img_funcs

import tensorflow as tf
from sklearn.model_selection import KFold
import random
import numpy as np
import PIL

from pathlib import Path
import cv2

#--- ----- CNN Implementation

### Controlled Randomization with a given seed

tf.random.set_seed(mp.SEED)
np.random.seed(mp.SEED)
random.seed(mp.SEED)


### Training + Validation Sets

# get training directory wrt current working directory
train_dataset_directory = Path.cwd() / 'train_imgdata' / 'trueclass_240x240_sortbyclass_full'

# create datapoints from dataset directory and label_name list
train_datapoints = img_funcs.create_datapoints_from_directory(
    train_dataset_directory.__str__(),
    mp.label_names_full,
    normalize=True
)
random.shuffle(train_datapoints)

# create k folds of training-validation splits
kf = KFold(n_splits=mp.FOLDS, shuffle=True)
    # KFold follows the numpy seed
train_datapoints_fold_indices = kf.split(train_datapoints)
    # kf.split only returns the randomized indices, not the actual sublists

# cleanup


### CNN Training

for f,current_splits in enumerate(train_datapoints_fold_indices):
    print('\n=== FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - start ===')

    # get training set datapoints from randomly generated indices
    current_trainset_indices = current_splits[0]
    current_trainset_points = [train_datapoints[i] for i in current_trainset_indices]

    # get validation set datapoints from randomly generated indices
    current_valset_indices = current_splits[1]
    current_valset_points = [train_datapoints[i] for i in current_valset_indices]

    # split training set into image and label lists
    current_train_images = np.array([point[0] for point in current_trainset_points])
    current_train_labels = [point[1] for point in current_trainset_points]
    current_train_labels = np.array(current_train_labels)

    # split validation set into image and label lists
    current_val_images = [point[0] for point in current_valset_points]
    current_val_images = np.array(current_val_images)
    current_val_labels = [point[1] for point in current_valset_points]
    current_val_labels = np.array(current_val_labels)

    # ---
    
    # only do training augmentation here, so it will happen FOLDSx total
    # in order to augment training data but not the validation data, per fold
    # https://stats.stackexchange.com/questions/482787/how-to-do-data-augmentation-and-cross-validation-at-the-same-time

    # ---

    model = tf.keras.models.Sequential()

    # initialize model architecture parameters
    model.add(tf.keras.layers.Conv2D(
        32, (3, 3),
        activation=mp.ACTIVATION,
        input_shape=(mp.img_HEIGHT, mp.img_WIDTH, 3)
    ))
    
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu') )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    
    # todo: model.add() a lot of shtuff

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation=mp.ACTIVATION))
    model.add(tf.keras.layers.Dense(mp.NUM_CLASSES))
       
    model.compile(
        optimizer=mp.OPTIMIZER,
        loss=mp.LOSS,
        metrics=mp.EVALUATION_METRICS
    )

    model.fit(
        current_train_images,
        current_train_labels,
        validation_data=(current_val_images, current_val_labels),
        epochs = 1,
        batch_size = mp.BATCH_SIZE
    )

    model.summary()

    print('=== FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - end ===\n')

#---

### Test Set

##test_images_DIR = os.path.join('test_imgdata', 'satellite_480x480')
##test_truemasks_DIR = os.path.join('test_imgdata', 'truemask_480x480')
##
##test_images = img_funcs.get_all_images_rgb(test_images_DIR)
##test_truemasks_color = img_funcs.get_all_images_rgb(test_truemasks_DIR)
##test_truemasks_index = [image_rgb_to_index(img) for img in test_truemasks_color]
##
##test_dataset = list(zip(test_images, test_images))
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
