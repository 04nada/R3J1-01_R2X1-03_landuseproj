import main_params3 as mp
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

# --- -----

### CNN Training

models = []
histories = []

mnist_train = tf.keras.preprocessing.image_dataset_from_directory(
    str(mp.ROOT_DIRECTORY / 'datasets' / 'mnist_png' / 'training'),
    shuffle = True,
    
    labels = "inferred",
    label_mode = "int",
    class_names = mp.label_names,

    image_size = (mp.img_HEIGHT, mp.img_WIDTH),
    batch_size = mp.BATCH_SIZE
).prefetch(
    buffer_size = tf.data.experimental.AUTOTUNE
)

for f in range(mp.FOLDS):  
    # first check if CHOSEN_FOLD is set to a specific acceptable value
    #     (otherwise, set CHOSEN_FOLD to -1)

    # if CHOSEN_FOLD is not -1, then only train a model for that specific fold,
    #     and skip all other folds in training
    
    if mp.CHOSEN_FOLD is not None:
        if f+1 == mp.CHOSEN_FOLD:
            print('\n=== CHOSEN: FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - start ===')
        else:
            continue
    else:
        print('\n=== FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - start ===')
    
    # ---

    model = mp.create_model()

    print('--- FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - model.summary() ---')
    model.summary()

    print('--- FOLD ' + str(f+1) + ' of ' + str(mp.FOLDS) + ' - model.fit() ---')
    history = model.fit(
        mnist_train,
        #validation_data = current_val_dataset,
        shuffle = True,
        
        epochs = mp.EPOCHS,
        batch_size = mp.BATCH_SIZE,
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                monitor = 'loss',
                filepath = str(
                    Path(mp.TRAINED_MODELS_DIRECTORY)
                    / ('mnist_model' + str(f).zfill(2) + '_{epoch:02d}.hdf5')
                ),
                save_weights_only = True,
                save_best_only = True,
                verbose = 1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor = 'loss',
                min_delta = mp.CALLBACK_MIN_DELTA,
                patience = mp.CALLBACK_PATIENCE,
                verbose = 1
            ),
            tf.keras.callbacks.History()
        ],
        
        verbose = 2
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
