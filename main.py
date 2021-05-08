import main_params as mp
import model_functions as model_funcs

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
train_dataset_directory = Path.cwd() / 'train_imgdata' / 'trueclass_240x240_sortbyclass_actual'

# create datapoint generators from dataset directory and label_name list
# use the ReGenerator class for reusability
train_datapoints_regen = model_funcs.ReGenerator(
    model_funcs.dataset_generator,
    (train_dataset_directory.__str__(),
        mp.label_names),
    {'normalize': True,
        'n': mp.TRAIN_SAMPLES_PER_CLASS}
)

# create k folds of random/shuffled training-validation splits
# KFold follows the numpy seed
kf = KFold(n_splits=mp.FOLDS, shuffle=True)

# less memory-exhaustive list with the same "length" as the datapoints generator
train_datapoints_length = train_datapoints_regen.length()
train_datapoints_filler_list = [i for i in range(train_datapoints_length)]
# kf.split only returns the randomized indices, not the actual sublists, as a generator of array pairs
train_datapoints_fold_indices = kf.split(train_datapoints_filler_list)


### CNN Training

models = []

for f in range(mp.FOLDS):
    print('\n=== FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - start ===')

    # get indices of training data and validation data for the current fold
    current_trainset_indices, current_valset_indices = next(train_datapoints_fold_indices)

    # get training set datapoints from randomly generated indices
    current_trainset_points = model_funcs.ReGenerator(
        lambda : (datapoint
            for i,datapoint in enumerate(train_datapoints_regen.gen((), {'log_progress': False}))
            if i in current_trainset_indices)
    )

    # get validation set datapoints from randomly generated indices
    current_valset_points = model_funcs.ReGenerator(
        lambda : (datapoint
            for i,datapoint in enumerate(train_datapoints_regen.gen((), {'log_progress': False}))
            if i in current_valset_indices)
    )

    # split training set into image and label generators
    current_train_images = model_funcs.ReGenerator(
        lambda : (point[0] for point in current_trainset_points.gen())
    )
    current_train_labels = model_funcs.ReGenerator(
        lambda : (point[1] for point in current_trainset_points.gen())
    )

    # split validation set into image and label generators
    current_val_images = model_funcs.ReGenerator(
        lambda : (point[0] for point in current_valset_points.gen())
    )
    current_val_labels = model_funcs.ReGenerator(
        lambda : (point[1] for point in current_valset_points.gen())
    )
    
    # ---
    
    # only do training data augmentation here, so it will happen FOLDSx total
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

    # continue applying convolutional layers while occasionally doing pooling
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=mp.ACTIVATION) )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=mp.ACTIVATION))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=mp.ACTIVATION))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=mp.ACTIVATION))
    
    # flatten CNN model to a single array of values
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation=mp.ACTIVATION))

    # final layer corresponds to the total number of classes for classifying into
    model.add(tf.keras.layers.Dense(mp.NUM_CLASSES))

    # compile model using specified tools and metrics
    print('--- FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - model.compile() ---')
    model.compile(
        optimizer=mp.OPTIMIZER,
        loss=mp.LOSS,
        metrics=mp.EVALUATION_METRICS
    )

    # convert final image and label generators to numpy arrays,
    # so that they get accepted as the model.fit() parameters
    print('1')
    current_train_images_array = np.array([image for image in current_train_images.gen()])
    print('2')
    current_train_labels_array = np.array([label for label in current_train_labels.gen()])
    print('3')
    current_val_images_array = np.array([image for image in current_val_images.gen()])
    print('4')
    current_val_labels_array = np.array([image for image in current_val_labels.gen()])

    # fit training and validation to model
    # also setting other parameters for how the model runs, namely epochs and batch size
    print('--- FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - model.fit() ---')
    model.fit(
        current_train_images_array,
        current_train_labels_array,
        validation_data=(current_val_images_array current_val_labels_array),
        epochs = 1,
        batch_size = mp.BATCH_SIZE
    )

    model.summary()
    #models.append(model)
    
    print('=== FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - end ===')

#---

### Test Set

# get testing directory wrt current working directory
test_dataset_directory = Path.cwd() / 'test_imgdata' / 'trueclass_240x240_sortbyclass_actual'

test_datapoints_regen = model_funcs.ReGenerator(
    model_funcs.dataset_generator,
    (test_dataset_directory.__str__(),
        mp.label_names),
    {'normalize': True,
        'n': mp.TEST_SAMPLES_PER_CLASS}
)

##test_images = model_funcs.get_all_images_rgb(test_images_DIR)
##test_truemasks_color = model_funcs.get_all_images_rgb(test_truemasks_DIR)
##test_truemasks_index = [image_rgb_to_index(img) for img in test_truemasks_color]
##
