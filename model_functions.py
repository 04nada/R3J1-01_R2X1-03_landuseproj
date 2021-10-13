import tensorflow as tf
import numpy as np

#--- ----- Logging

def t_print(text:str, toggle:bool) -> None:
    if toggle:
        print(text)

#--- ----- Image Displaying

# display image/s using matplotlib.pyplot
# https://www.tensorflow.org/tutorials/images/segmentation

from matplotlib import pyplot as plt

def display_images(display_list:list) -> None:
    plt.figure(figsize=(15, 15))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        #plt.axis('off')
    plt.show()

def print_image(image:list) -> None:
    for row_of_pixels in image:
        print(row_of_pixels)

# --- ----- File Processing

from pathlib import Path

VALID_IMAGE_EXTENSIONS = [
    '.jpg',
    '.jpeg',
    '.png',
]

# --- ----- Dataset Generation

# ReGenerator class to allow generator reusability
class ReGenerator:
    def __init__(self, gen_f:'generator_function',
    args:'iterator'=(),
    kwargs:'dict'={}):
        self.generator_function = gen_f
        self.generator_args = tuple(args)
        self.generator_kwargs = kwargs

    def gen(self,
    extra_args:'iterator'=(),
    extra_kwargs:'dict'={}) -> 'generator':
        # unpack main parameters first (*args)
        # then unpack named parameters at the end (**kwargs)
        return self.generator_function(*self.generator_args, *extra_args, **self.generator_kwargs, **extra_kwargs)

    def length(self,
    extra_args:'iterator'=(),
    extra_kwargs:'dict'={}) -> int:
        # if length has been previously computed, return that memorized value
        if hasattr(self, 'generator_length'):
            return self.generator_length

        # get the total number of values that the generator function produces
        genlen = 0
        for g in self.gen(*extra_args, **extra_kwargs):
            genlen += 1

        # remember the length if it has not been computed before
        self.generator_length = genlen
        
        return genlen

# ---

import random
from PIL import Image

def generate_rgb_image_from_path(image_filepath:str) -> 'matrix':
    # read image file data as a matrix of RGB pixels
    image = Image.open(image_filepath).convert('RGB')
    image_matrix = np.array(image)

    return image_matrix

def get_dataset_size(dataset_dirpath:str):
    dataset_dir_path = Path(dataset_dirpath)

    dataset_size = 0

    # count all images in dataset folder recursively
    for sub_path in dataset_dir_path.rglob('*'):
        if sub_path.is_file() and sub_path.suffix in VALID_IMAGE_EXTENSIONS:
            dataset_size += 1

    return dataset_size

def dataset_generator(dataset_dirpath:str, label_names:list,
*, normalize:bool=False, n=None, log_progress:bool=True) -> 'generator':
    t_print('=== Yield Dataset: from Directory - start ===', log_progress)
    
    # get dataset directory as Path object
    dataset_dir_path = Path(dataset_dirpath)

    # get all class directories found inside the dataset directory
    # iterdir() to check all subpaths inside a Path object
    class_dir_paths = (i for i in dataset_dir_path.iterdir() if i.is_dir())
    
    for c_d_p,class_dir_path in enumerate(class_dir_paths):
        t_print('--- CD_D: ' + class_dir_path.name + ' - ' + str(c_d_p) + ' of ' + str(len(label_names)) + ' classes finished ---', log_progress)

        # get label number of current dataclass path
        class_name = class_dir_path.name
        label_number = label_names.index(class_name)

        # generate each datapoint individually
        # "yield from" is used to keep yielding values from another generator
        class_dirpath = class_dir_path.__str__()
        
        yield from yield_datapoints_rgb(class_dirpath, label_number, normalize=normalize, n=n)

    t_print('--- CD_D: ' + str(len(label_names)) + ' of ' + str(len(label_names)) + ' classes finished---', log_progress)

    t_print('=== Yield Dataset: from Directory - finish ===', log_progress)

def yield_datapoints_rgb(image_dirpath:str, label_number:int,
*, normalize:bool=False, n=None) -> 'generator':
    # get image directory as Path object
    image_dir_path = Path(image_dirpath)

    # get all valid image files inside the given image directory
    # use the ReGenerator class to get length of generator
    image_file_paths = ReGenerator(
        lambda : (subfile_path for subfile_path in image_dir_path.iterdir() if subfile_path.is_file() and subfile_path.suffix in VALID_IMAGE_EXTENSIONS)
    )
    
    # if n is specified, randomly select n indices from the subfiles
    if n is not None:
        random_indices = random.sample(range(image_file_paths.length()), n)

    # create an (image, label) datapoint for each valid image file obtained
    for i_f_p,image_file_path in enumerate(image_file_paths.gen()):
        # if n is specified, do not yield an image if
        #     it is not within the randomly generated indices
        if n is not None and i_f_p not in random_indices:
            continue

        # convert image filepath from Path object into its string
        image_filepath = image_file_path.__str__()
        image = generate_rgb_image_from_path(image_filepath)

        # if normalize is True, pixel values get converted from [0,255] to [0,1]
        if normalize:
            image = image / 255.0

        # ---
        
        yield (image, label_number)

#---

##define dataAugmentation:
##    let x be a random number from [0,1]
##    if > 0.5:
##        flip the image
##    else:
##        rotate the image by a random degree
##

def augment_data(image:list, mask:list):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(image)
        input_mask = tf.image.flip_left_right(mask)
        pass

    # todo : clarify if do all 4 rotations from 0 to 270?
    # still needs fixing with regards to actually increasing the number of images

#--- ----- Model Graphs

def plot_training_graphs(number_of_epochs:int,
**kwargs) -> None:
    print('Plotting Training Graphs...')

    # ---
    
    epochs = range(1, number_of_epochs+1)

    training_accuracy = kwargs.get('train_accuracy', None)
    training_precision = kwargs.get('train_precision', None)
    training_recall = kwargs.get('train_recall', None)

    validation_accuracy = kwargs.get('val_accuracy', None)
    validation_precision = kwargs.get('val_precision', None)
    validation_recall = kwargs.get('val_recall', None)

    training_loss = kwargs.get('train_loss', None)
    validation_loss = kwargs.get('val_loss', None)

    # --- Metric Graphs

    if training_accuracy is not None:
        print(training_accuracy)
        plt.plot(epochs, training_accuracy, 'ro-', label='Training Accuracy')               # red circle + solid line (line graph)

    if training_precision is not None:
        plt.plot(epochs, training_precision, 'go-', label='Training Precision')             # green circle + solid line (line graph)

    if training_recall is not None:
        plt.plot(epochs, training_recall, 'bo-', label='Training Recall')                   # blue circle + solid line (line graph)

    if validation_accuracy is not None:
        plt.plot(epochs, validation_accuracy, 'ro--', label='Validation Accuracy')          # cyan circle + dashed line (line graph)

    if validation_precision is not None:
        plt.plot(epochs, validation_precision, 'go--', label='Validation Precision')        # magenta circle + dashed line (line graph)

    if validation_recall is not None:
        plt.plot(epochs, validation_recall, 'bo--', label='Validation Recall')              # yellow circle + dashed line (line graph)

    plt.title('Model Metrics')
    plt.legend()

    # --- Loss Graphs

    plt.figure()

    if training_loss is not None:
        training_loss = list(training_loss)
        plt.plot(epochs, training_loss, 'ko-', label='Training Loss')                       # black circle + solid line (line graph)

    if validation_loss is not None:
        plt.plot(epochs, validation_loss, 'ko--', label='Validation Loss')                  # black circle + dashed line (line graph)

    plt.title('Model Loss')
    plt.legend()
    
    # ---

    plt.show()

def plot_test_graphs(number_of_epochs:int,
**kwargs) -> None:
    print('Plotting Test Graphs...')

    # ---
    
    epochs = range(1, number_of_epochs+1)

    model_accuracy = kwargs.get('test_accuracy', None)
    model_precision = kwargs.get('test_precision', None)
    model_recall = kwargs.get('test_recall', None)
    model_F1score = kwargs.get('test_F1score', None)

    model_loss = kwargs.get('test_loss', None)

    # --- Metric Graphs

    if model_accuracy is not None:
        plt.plot(epochs, model_accuracy, 'ro-', label="Accuracy")       # red circle + solid line (line graph

    if model_precision is not None:
        plt.plot(epochs, model_precision, 'go-', label="Precision")     # green circle + solid line (line graph)

    if model_recall is not None:
        plt.plot(epochs, model_recall, 'bo-', label="Recall")           # blue circle + solid line (line graph)

    if model_F1score is not None:
        plt.plot(epochs, model_F1score, 'mo-', label="F1 Score")        # magenta circle + solid line (line graph)

    plt.title('Model Metrics')
    plt.legend()

    # --- Loss Graph

    plt.figure()

    if model_loss is not None:
        plt.plot(epochs, model_loss, 'ko-', label="Loss")                     # black circle + solid line (line graph)

    plt.title("Model Loss")
    plt.legend()

    plt.show()

# --- ----- Random Index Partitioning

# standalone KFold implementation
# values determined by the 'random' package
def random_index_partition(num_groups, num_elements):
    indices = []
    
    unused_indices = [i for i in range(num_elements)]
    random.shuffle(unused_indices)

    # check how many (E) excess elements there will be after gett
    excess_elements = num_elements % num_groups

    for i in range(num_groups):
        current_num_elements = num_elements // num_groups

        # if there are E excess elements from modulo, then the first E groups will have an extra element
        if i < excess_elements:
            current_num_elements += 1

        # ---
    
        chosen_indices = unused_indices[0:current_num_elements]
        
        indices.append(chosen_indices)
        unused_indices = unused_indices[current_num_elements:]

    return indices
