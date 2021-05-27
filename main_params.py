import tensorflow as tf
from pathlib import Path

#--- ----- CNN Parameters

### image parameters

img_COLORMAP_HEIGHT = 20
img_COLORMAP_WIDTH = 20

img_HEIGHT = 240
img_WIDTH = 240

lookup_rgb_to_index_full = {
    (97,64,31): 0,	    # agricultural - brown - #61401F
    (160,32,239): 1,	    # commercial - purple - #A020EF
    (0,0,254): 2,           # harbor_seawater - blue - #0000FE
    (221,190,170): 3,       # industrial - beige - #DDBEAA
    (237,0,0): 4,   	    # institutional - red - #ED0000
    (45,137,86): 5,	    # recreational - green - #2D8956
    (254,165,0): 6,	    # residential - yellow - #FEA500
    (0,0,87): 7	            # transport - dark blue - #000057
}

lookup_rgb_to_index = {
    (160,32,239): 0,	    # commercial - purple - #A020EF
    (221,190,170): 1,       # industrial - beige - #DDBEAA
    (237,0,0): 2,   	    # institutional - red - #ED0000
    (45,137,86): 3,	    # recreational - green - #2D8956
    (254,165,0): 4,	    # residential - yellow - #FEA500
    (0,0,87): 5		    # transport - dark blue - #000057
}

label_names_full = [
    'agricultural',
    'commercial',
    'harbor_seawater',
    'industrial',
    'institutional',
    'recreational',
    'residential',
    'transport'
]

label_names = [
    'commercial',
    'industrial',
    'institutional',
    'recreational',
    'residential',
    'transport'
]

#---

### model interation parameters

SEED = 727                                          # consistent randomization from a set seed

NUM_CLASSES = 6

BATCH_SIZE = 32                                     # power of 2 for optimized CPU/GPU usage

# training set
TRAIN_DATASET_DIRECTORY = Path.cwd() / 'train_imgdata' / 'trueclass_240x240_sortbyclass_actual'

TRAIN_SAMPLES_PER_CLASS = 400
TRAIN_SIZE = TRAIN_SAMPLES_PER_CLASS * NUM_CLASSES
TRAIN_STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE    # floor division

FOLDS = 5

# test set
TEST_DATASET_DIRECTORY = Path.cwd() / 'test_imgdata' / 'trueclass_240x240_sortbyclass_actual'

TEST_SAMPLES_PER_CLASS = 1
TEST_SIZE = TEST_SAMPLES_PER_CLASS * NUM_CLASSES
TEST_STEPS_PER_EPOCH = TEST_SIZE // BATCH_SIZE      # floor division

EPOCHS = 5                                          # filler number, just has to be more than enough to overfit before reaching the final epoch

#---

### model implementation parameters

ACTIVATION = 'relu'

OPTIMIZER = 'sgd'                                           # Stochastic Gradient Descent
LOSS = tf.keras.losses.SparseCategoricalCrossentropy()      # Sparse Categorical Cross-Entropy
EVALUATION_METRICS = [
    tf.keras.metrics.SparseCategoricalCrossentropy(),
    tf.keras.metrics.Accuracy(),
    tf.keras.metrics.Precision()
]
