import main_params3 as mp
import model_functions as model_funcs

import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt
import random
import numpy as np

from pathlib import Path

# --- ----

model_file_paths = [sub_file for sub_file in Path(mp.TRAINED_MODELS_DIRECTORY).iterdir() if sub_file.is_file() and sub_file.suffix == '.hdf5']
number_of_models = len(model_file_paths)

print('Detected ' + str(number_of_models) + ' models in directory:')
for i,model_file_path in enumerate(model_file_paths):
    print('(' + str(i) + ') ' + str(model_file_path))

model_index = int(input('\nWhich model will be evaluated? [0-' + str(number_of_models - 1) + ']: '))
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

### Test Set

##if mp.CHOSEN_FOLD > 0:
##    test_datapoints_regen = model_funcs.ReGenerator(
##        model_funcs.dataset_generator,
##        (mp.TEST_DATASET_DIRECTORY.__str__(),
##            mp.label_names),
##        {'normalize': True,
##            'n': None}
##    )
##
##    test_images = model_funcs.ReGenerator(
##        lambda : (point[0] for point in test_datapoints_regen.gen())
##    )
##
##    test_labels = model_funcs.ReGenerator(
##        lambda : (point[1] for point in test_datapoints_regen.gen())
##    )
##
##    test_images_array = np.array([image for image in test_images.gen()])
##    test_labels_array = np.array([label for label in test_labels.gen()])
##
##    # ---
##    

a = model_funcs.generate_rgb_image_from_path(
    Path(mp.TRAIN_DATASET_DIRECTORY)
    / '0'
    / '1.png'
)

results = test_model.evaluate(
    test_dataset,
    verbose=2
)

##confusion_matrix = model_funcs.generate_confusion_matrix(
##    test_model,
##    mp.TRAIN_DATASET_DIRECTORY,
##    mp.label_names
##)
