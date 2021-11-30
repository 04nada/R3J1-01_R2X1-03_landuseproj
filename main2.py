import main_params as mp
import model_functions as model_funcs

import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt
import random
import numpy as np

from pathlib import Path

#--- ----- CNN Implementation

### Controlled Randomization with a given seed

tf.random.set_seed(mp.SEED)
np.random.seed(mp.SEED)
random.seed(mp.SEED)


# ---

### Training + Validation Sets, using Fold folders

train_datasets = [
    tf.keras.preprocessing.image_dataset_from_directory(
        str(Path(mp.TRAIN_DATASET_DIRECTORIES2[f]) / 'training'),
        image_size = (mp.img_HEIGHT, mp.img_WIDTH),
        batch_size = mp.BATCH_SIZE
    ).prefetch(
        buffer_size = tf.data.experimental.AUTOTUNE
    ) for f in range(mp.FOLDS)
]

val_datasets = [
    tf.keras.preprocessing.image_dataset_from_directory(
        str(Path(mp.TRAIN_DATASET_DIRECTORIES[f]) / 'validation'),
        image_size = (mp.img_HEIGHT, mp.img_WIDTH),
        batch_size = mp.BATCH_SIZE
    ).prefetch(
        buffer_size = tf.data.experimental.AUTOTUNE
    ) for f in range(mp.FOLDS)
]

# --- -----

### CNN Training

models = []
histories = []

for f in range(mp.FOLDS):
    # first check if CHOSEN_FOLD is set to a specific acceptable value
    #     (otherwise, set CHOSEN_FOLD to -1)

##    # if CHOSEN_FOLD is not -1, then only train a model for that specific fold,
##    #     and skip all other folds in training
##    
##    if mp.CHOSEN_FOLD > 0:
##        if f+1 == mp.CHOSEN_FOLD:
##            print('\n=== CHOSEN: FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - start ===')
##        else:
##            continue
##    else:
##        print('\n=== FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - start ===')
##    
##    # ---
##    
##    # only do training data augmentation here, so it will happen FOLDSx total
##    # in order to augment training data but not the validation data, per fold
##    # https://stats.stackexchange.com/questions/482787/how-to-do-data-augmentation-and-cross-validation-at-the-same-time

    # ---

    model = mp.create_model()

    print('--- FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - model.summary() ---')
    model.summary()

    print('--- FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - model.fit() ---')
    history = model.fit(
        train_datasets[f],
        validation_data = val_datasets[f],
        shuffle = True,
        
        epochs = mp.EPOCHS,
        batch_size = mp.BATCH_SIZE,
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath = str(
                    Path(mp.TRAIN_CHECKPOINTS_DIRECTORY)
                    / ('model' + str(f).zfill(2) + '_{epoch:02d}.hdf5')
                ),
                save_weights_only = True,
                save_best_only = True,
                verbose = 1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor = 'val_loss',
                min_delta = mp.CALLBACK_MIN_DELTA,
                patience = mp.CALLBACK_PATIENCE,
                verbose = 1
            ),
            tf.keras.callbacks.History()
        ],
        
        verbose = 1
    )

    accuracy = history.history['accuracy']
    loss = history.history['loss']

    actual_num_epochs = len(loss)

    model_funcs.plot_training_graphs(
        actual_num_epochs,
        train_accuracy = accuracy,
        train_loss = loss
    )

    # ---

    models.append(model)
    histories.append(history.history)
    
    break

    # ---

    # check for which end-of-fold message to print
    
    if mp.CHOSEN_FOLD > 0:
        print('=== CHOSEN FOLD: Fold ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - end ===')
        break
    else:
        print('=== FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - end ===')
