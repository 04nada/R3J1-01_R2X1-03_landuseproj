import main_params as mp
import model_functions as model_funcs

import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt
import random
import numpy as np

from pathlib import Path
import pickle

# --- ----

models_dir_path = Path(mp.TRAINED_MODELS_DIRECTORY)

# ---

model_file_paths = [sub_file for sub_file in models_dir_path.iterdir() if sub_file.is_file() and sub_file.suffix == '.hdf5']
number_of_models = len(model_file_paths)

print('Detected ' + str(number_of_models) + ' models in directory:')
for i,model_file_path in enumerate(model_file_paths):
    print('(' + str(i) + ') ' + str(model_file_path))

#model_index = int(input('\nWhich model will be evaluated? [0-' + str(number_of_models - 1) + ']: '))

# ---

a = [5, 6, 7, 8, 9, 10,
     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
     41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
     51, 52, 53, 54, 55, 56, 57, 58, 59, 60]

for model_index in a:
    chosen_model = model_file_paths[model_index]

    # ---

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        mp.TEST_DATASET_DIRECTORY,
        shuffle = True,
                
        labels = "inferred",
        label_mode = "int",
        class_names = mp.label_names,
         
        image_size = (mp.img_HEIGHT, mp.img_WIDTH),
        batch_size = mp.BATCH_SIZE
    ).prefetch(
        buffer_size = tf.data.experimental.AUTOTUNE
    )

    # ---

    test_model = mp.create_model()
    test_model.load_weights(chosen_model)

    test_model.summary()

    # ---

    results = test_model.evaluate(
        test_dataset,
        verbose=2
    )

    results_filename = str(Path(chosen_model).stem).replace('history', 'results')
    pickle.dump(results, open(
        str(Path(mp.TESTING_RESULTS_DIRECTORY)
            / (str(Path(chosen_model).stem) + '__results.obj')
        ), 'wb')
    )

    # ---

    fold = int(results_filename[results_filename.find('fold')+4:results_filename.find('fold')+6])

    train_confusion_matrix = model_funcs.generate_confusion_matrix(
        test_model,
        str(Path(mp.TRAIN_DATASET_DIRECTORIES[fold-1]) / 'training'),
        mp.label_names,
        size = (mp.img_WIDTH, mp.img_HEIGHT)
    )
    pickle.dump(train_confusion_matrix, open(
        str(Path(mp.TRAINING_HISTORIES_DIRECTORY)
            / (str(Path(chosen_model).stem) + '__train_confusion_matrix.obj')
        ), 'wb')
    )

    # ---

    val_confusion_matrix = model_funcs.generate_confusion_matrix(
        test_model,
        str(Path(mp.TRAIN_DATASET_DIRECTORIES[fold-1]) / 'validation'),
        mp.label_names,
        size = (mp.img_WIDTH, mp.img_HEIGHT)
    )
    pickle.dump(val_confusion_matrix, open(
        str(Path(mp.TRAINING_HISTORIES_DIRECTORY)
            / (str(Path(chosen_model).stem) + '__val_confusion_matrix.obj')
        ), 'wb')
    )

# ---

##test_confusion_matrix = model_funcs.generate_confusion_matrix(
##    test_model,
##    mp.TEST_DATASET_DIRECTORY,
##    mp.label_names,
##    size = (mp.img_WIDTH, mp.img_HEIGHT)
##)
##pickle.dump(test_confusion_matrix, open(
##    str(Path(mp.TESTING_RESULTS_DIRECTORY)
##        / (str(Path(chosen_model).stem) + '__test_confusion_matrix.obj')
##    ), 'wb')
##)
