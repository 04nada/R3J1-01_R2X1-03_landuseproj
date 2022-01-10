import tensorflow as tf
import tensorflow_addons as tfa
import tf_metrics
from pathlib import Path

ROOT_DIRECTORY = Path.cwd()

#--- ----- CNN Parameters

### image parameters

img_HEIGHT = 28
img_WIDTH = 28

label_names = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9'
]

#--- ----- CNN Training

### model interaction parameters

SEED = 727                                          # consistent randomization from a set seed

NUM_CLASSES = 10
FOLDS = 5

EPOCHS = 1                                          # filler number, just has to be more than enough to overfit before reaching the final epoch
BATCH_SIZE = 16                                     # power of 2 for optimized CPU/GPU usage
LEARNING_RATE = 0.01                                # decimal power of 10

# training set
TRAIN_DATASET_DIRECTORY = str(
    ROOT_DIRECTORY
    / 'datasets'
    / 'mnist_png'
    / 'training'
)

CALLBACK_MIN_DELTA = 0
CALLBACK_PATIENCE = 2

#--- ----- CNN Testing

CHOSEN_FOLD = 1

# test set
TEST_DATASET_DIRECTORY = str(
    ROOT_DIRECTORY
    / 'datasets'
    / 'mnist_png'
    / 'testing'
)

#---

TRAINED_MODELS_DIRECTORY = str(
    ROOT_DIRECTORY
    / 'models'
)

#---

### custom metrics

class CategoricalTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        values = tf.cast(values, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)

### model implementation parameters

ACTIVATION = 'relu'

OPTIMIZER = tf.keras.optimizers.SGD(                                       # Stochastic Gradient Descent
    learning_rate = LEARNING_RATE
)
LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)      # Sparse Categorical Cross-Entropy
EVALUATION_METRICS = [
    'accuracy',
    tf.keras.metrics.SparseCategoricalAccuracy(),
    CategoricalTruePositives()
#    tf_metrics.recall()
]

### model creation

def create_model():
    model = tf.keras.models.Sequential()

    # initialize model architecture parameters
    model.add(tf.keras.layers.Conv2D(
        32, (3, 3),
        activation=ACTIVATION,
        input_shape=(img_HEIGHT, img_WIDTH, 3)
    ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # continue applying convolutional layers while occasionally doing pooling
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=ACTIVATION))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation=ACTIVATION))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    # flatten CNN model to a single array of values
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation=ACTIVATION))

    # final layer corresponds to the total number of classes for classifying into
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid"))

    # compile model using specified tools and metrics
    #print('--- FOLD ' + str(f+1) + ' of ' + str(FOLDS) + ' - model.compile() ---')
    model.compile(
        optimizer=OPTIMIZER,
        loss=LOSS,
        metrics=EVALUATION_METRICS
    )
    tf.keras.backend.set_value(model.optimizer.learning_rate, LEARNING_RATE)

    return model

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(CollectOutputAndTarget, self).__init__()
        self.targets = []  # collect y_true batches
        self.outputs = []  # collect y_pred batches

        # the shape of these 2 variables will change according to batch shape
        # to handle the "last batch", specify `validate_shape=False`
        self.var_y_true = tf.Variable(0., validate_shape=False)
        self.var_y_pred = tf.Variable(0., validate_shape=False)

    def on_batch_end(self, batch, logs=None):
        # evaluate the variables and save them into lists
        self.targets.append(K.eval(self.var_y_true))
        self.outputs.append(K.eval(self.var_y_pred))


