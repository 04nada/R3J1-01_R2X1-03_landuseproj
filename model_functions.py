import tensorflow as tf

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


#--- ----- Image Processing

from pathlib import Path
import cv2

valid_image_extensions = [
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

def dataset_generator(dataset_path:str, label_names:list,
*, normalize:bool=False, n=None, log_progress:bool=True) -> 'generator':
    t_print('=== Yield Dataset: from Directory - start ===', log_progress)

    # get dataset directory as Path object
    dataset_path_obj = Path(dataset_path)

    # get all class directories found inside the dataset directory
    # iterdir() to check all subpaths inside a Path object
    dataclasses_paths = (i for i in dataset_path_obj.iterdir() if i.is_dir())
    
    for sp,subpath_obj in enumerate(dataclasses_paths):
        t_print('--- CD_D: ' + subpath_obj.name + ' - ' + str(sp) + ' of ' + str(len(label_names)) + ' classes finished ---', log_progress)

        # get label number of current dataclass path
        image_dataclass_name = subpath_obj.name
        label_number = label_names.index(image_dataclass_name)

        # generate each datapoint individually
        # "yield from" is used to keep yielding values from another generator
        subpath = subpath_obj.__str__()
        yield from yield_datapoints_rgb(subpath, label_number, normalize=normalize, n=n)

    t_print('--- CD_D: ' + str(len(label_names)) + ' of ' + str(len(label_names)) + ' classes finished---', log_progress)

    t_print('=== Yield Dataset: from Directory - finish ===', log_progress)

def yield_datapoints_rgb(image_directory:str, label_number:int,
*, normalize:bool=False, n=None) -> 'generator':
    # get image directory as Path object
    image_directory_obj = Path(image_directory)

    # get all valid image files inside the given image directory
    # use the ReGenerator class to get length of generator
    image_files = ReGenerator(
        lambda : (subpath for subpath in image_directory_obj.iterdir() if subpath.is_file() and subpath.suffix in valid_image_extensions)
    )
    
    # if n is specified, randomly select n indices from the subfiles
    if n is not None:
        random_indices = random.sample(range(image_files.length()), n)

    # create an (image, label) datapoint for each valid image file obtained
    for i_f,image_file in enumerate(image_files.gen()):
        # if n is specified, do not yield an image if
        #     it is not within the randomly generated indices
        if n is not None and i_f not in random_indices:
            continue

        # --- Image Generation

        # convert image filepath from Path object into its string
        image_file_str = image_file.__str__()

        # read image file data as a matrix of RGB pixels
        image = cv2.imread(image_file_str)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
