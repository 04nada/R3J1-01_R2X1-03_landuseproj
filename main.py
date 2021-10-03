import main_params as mp
import model_functions as model_funcs

import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt

import random
import numpy as np

from pathlib import Path
import cv2

#--- ----- CNN Implementation

### Controlled Randomization with a given seed

tf.random.set_seed(mp.SEED)
np.random.seed(mp.SEED)
random.seed(mp.SEED)


### Training + Validation Sets

# create datapoint generators from dataset directory and label_name list
# use the ReGenerator class for reusability
train_datapoints_regen = model_funcs.ReGenerator(
    model_funcs.dataset_generator,
    (mp.TRAIN_DATASET_DIRECTORY.__str__(),
        mp.label_names),
    {'normalize': True,
        'n': None} # mp.TRAIN_SAMPLES_PER_CLASS
)

# partition number of datapoints into k groups, for k folds
train_datapoints_fold_indices = model_funcs.random_index_partition(
    mp.FOLDS,
    train_datapoints_regen.length()
)


### CNN Training

models = []

for f in range(mp.FOLDS):
    # first check if CHOSEN_FOLD is set to a specific acceptable value
    #     (otherwise, set CHOSEN_FOLD to -1)

    # if CHOSEN_FOLD is not -1, then only train a model for that specific fold,
    #     and skip all other folds in training
    
    if mp.CHOSEN_FOLD > 0:
        if f+1 == mp.CHOSEN_FOLD:
            print('\n=== CHOSEN: FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - start ===')
        else:
            continue
    else:
        print('\n=== FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - start ===')

    # ---
    
    # get indices of training set and validation set for the current fold
    current_trainset_indices = []
    current_valset_indices = []

    for i,group in enumerate(train_datapoints_fold_indices):
        if i == f:
            current_valset_indices.extend(group)
        else:
            current_trainset_indices.extend(group)

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
        16, (9, 9),
        activation=mp.ACTIVATION,
        input_shape=(mp.img_HEIGHT, mp.img_WIDTH, 3)
    ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # continue applying convolutional layers while occasionally doing pooling
    model.add(tf.keras.layers.Conv2D(32, (7, 7), activation=mp.ACTIVATION) )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(16, (5, 5), activation=mp.ACTIVATION) )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(32, (5, 5), activation=mp.ACTIVATION))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
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
    tf.keras.backend.set_value(model.optimizer.learning_rate, mp.LEARNING_RATE)

    print('--- FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - model.summary() ---')
    model.summary()

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
    history = model.fit(
        current_train_images_array,
        current_train_labels_array,
        validation_data = (current_val_images_array, current_val_labels_array),
        epochs = mp.EPOCHS,
        batch_size = mp.BATCH_SIZE,
        verbose = 2
    )

    # ---

    models.append(model)

##    pyplot.subplot(212)
##    pyplot.title('Accuracy')
##    pyplot.plot(history.history['accuracy'], label='train')
##    pyplot.plot(history.history['val_accuracy'], label='test')
##    pyplot.legend()
##    pyplot.show()

    # ---

    # check for which end-of-fold message to print
    
    if mp.CHOSEN_FOLD > 0:
        print('=== CHOSEN FOLD: Fold ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - end ===')
        break
    else:
        print('=== FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - end ===')

#---

### Test Set

if mp.CHOSEN_FOLD > 0:
    test_datapoints_regen = model_funcs.ReGenerator(
        model_funcs.dataset_generator,
        (mp.TEST_DATASET_DIRECTORY.__str__(),
            mp.label_names),
        {'normalize': True,
            'n': None}
    )

    test_images = model_funcs.ReGenerator(
        lambda : (point[0] for point in test_datapoints_regen.gen())
    )

    test_labels = model_funcs.ReGenerator(
        lambda : (point[1] for point in test_datapoints_regen.gen())
    )

    test_images_array = np.array([image for image in test_images.gen()])
    test_labels_array = np.array([label for label in test_labels.gen()])

    # ---
    
    results = models[0].evaluate(test_images_array, test_labels_array, verbose=2)
