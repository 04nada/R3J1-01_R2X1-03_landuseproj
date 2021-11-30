import tensorflow as tf
import tensorflow_addons as tfa
from pathlib import Path

ROOT_DIRECTORY = Path.cwd()

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

#--- ----- CNN Training

### model interaction parameters

SEED = 727                                          # consistent randomization from a set seed

NUM_CLASSES = 6
FOLDS = 5

EPOCHS = 5                                          # filler number, just has to be more than enough to overfit before reaching the final epoch
BATCH_SIZE = 16                                     # power of 2 for optimized CPU/GPU usage
LEARNING_RATE = 0.01                                # decimal power of 10

# training set
TRAIN_DATASET_DIRECTORY = str(
    ROOT_DIRECTORY
    / 'datasets'
    / 'train_images'
    / 'trueclass_240x240_sortbyclass_actual'
)
TRAIN_DATASET_DIRECTORIES = [
    str(
        ROOT_DIRECTORY
        / 'datasets'
        / 'train_images'
        / 'trueclass_240x240_sortbyclass_actual_folds'
        / 'folds'
        / ('fold'+str(i+1))
    )
    for i in range(FOLDS)
]

TRAIN_DATASET_DIRECTORIES2 = [
    str(
        ROOT_DIRECTORY
        / 'datasets'
        / 'train_images'
        / 'trueclass_240x240_sortbyclass_actual2_folds'
        / 'folds'
        / ('fold'+str(i+1))
    )
    for i in range(FOLDS)
]

TRAIN_SAMPLES_PER_CLASS = 400
TRAIN_SIZE = TRAIN_SAMPLES_PER_CLASS * NUM_CLASSES
TRAIN_STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE    # floor division

CALLBACK_MIN_DELTA = 0
CALLBACK_PATIENCE = 2

#--- ----- CNN Testing

CHOSEN_FOLD = 1

# test set
TEST_DATASET_DIRECTORY = str(
    ROOT_DIRECTORY
    / 'datasets'
    / 'test_images'
    / 'trueclass_240x240_sortbyclass_actual'
)

TEST_SAMPLES_PER_CLASS = 1
TEST_SIZE = TEST_SAMPLES_PER_CLASS * NUM_CLASSES
TEST_STEPS_PER_EPOCH = TEST_SIZE // BATCH_SIZE      # floor division

#---

TRAINED_MODELS_DIRECTORY = str(
    ROOT_DIRECTORY
    / 'models'
)

#---

### model implementation parameters

ACTIVATION = 'relu'

OPTIMIZER = 'sgd'                                                           # Stochastic Gradient Descent
LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)      # Sparse Categorical Cross-Entropy
EVALUATION_METRICS = [
    'accuracy'
]

### model creation

def create_model():
    model = tf.keras.models.Sequential()

    # initialize model architecture parameters
    model.add(tf.keras.layers.Conv2D(
        64, (9, 9),
        activation=ACTIVATION,
        input_shape=(img_HEIGHT, img_WIDTH, 3)
    ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # continue applying convolutional layers while occasionally doing pooling
    model.add(tf.keras.layers.Conv2D(32, (7, 7), activation=ACTIVATION) )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(32, (5, 5), activation=ACTIVATION) )
    model.add(tf.keras.layers.MaxPooling2D((3, 3)))
    model.add(tf.keras.layers.Conv2D(16, (5, 5), activation=ACTIVATION))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    # flatten CNN model to a single array of values
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation=ACTIVATION))

    # final layer corresponds to the total number of classes for classifying into
    model.add(tf.keras.layers.Dense(NUM_CLASSES))

    # compile model using specified tools and metrics
    #print('--- FOLD ' + str(f+1) + ' of ' + str(FOLDS) + ' - model.compile() ---')
    model.compile(
        optimizer=OPTIMIZER,
        loss=LOSS,
        metrics=EVALUATION_METRICS
    )
    tf.keras.backend.set_value(model.optimizer.learning_rate, LEARNING_RATE)

    return model
