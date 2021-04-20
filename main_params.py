import tensorflow as tf

#--- ----- CNN Parameters

### image parameters

img_COLORMAP_HEIGHT = 20
img_COLORMAP_WIDTH = 20

img_HEIGHT = 240
img_WIDTH = 240

lookup_rgb_to_index = {
    (97,64,31): 0,	    # agricultural - brown - #61401F
    (160,32,239): 1,	    # commercial - purple - #A020EF
    (221,190,170): 2,       # industrial - beige - #DDBEAA
    (237,0,0): 3,   	    # institutional - red - #ED0000
    (45,137,86): 4,	    # recreational - green - #2D8956
    (254,165,0): 5,	    # residential - yellow - #FEA500
    (0,0,87): 6		    # transport - dark blue - #000057
}

label_names = [
    'agricultural',
    'commercial',
    'industrial',
    'institutional',
    'recreational',
    'residential',
    'transport'
]

#---

### model interation parameters

SEED = 727                                      # consistent randomization from a set seed

TRAIN_SIZE = 2500                               
FOLDS = 5                                       
BATCH_SIZE = 32                                 # power of 2 for optimized CPU/GPU usage
STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE      # floor division

TEST_SIZE = 500                                 

EPOCHS = 1000                                   # filler number, just has to be more than enough to overfit before reaching the final epoch

#---

### model implementation parameters

ACTIVATION = 'relu'

OPTIMIZER = 'sgd'                                           # Stochastic Gradient Descent
##LOSS = tf.keras.losses.SparseCategoricalCrossentropy()      # Sparce Categorical Cross-Entropy
##EVALUATION_METRICS = [
##    tf.keras.metrics.MeanIoU()
##]
