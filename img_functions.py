import tensorflow as tf

#--- ----- Image Displaying

# display image/s using matplotlib.pyplot
# https://www.tensorflow.org/tutorials/images/segmentation

from matplotlib import pyplot as plt

def display_images(display_list:list):
    plt.figure(figsize=(15, 15))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        #plt.axis('off')
    plt.show()

def print_image(image: list):
    for row_of_pixels in image:
        print(row_of_pixels)

#--- ----- Image Processing

valid_image_extensions = [
    '.jpg',
    '.jpeg',
    '.png',
]

### create list of all images inside a folder path + its subfolders

from pathlib import Path
import cv2

def get_all_image_names(file_path:str):
    image_names = []

    # convert file_path to pathlib object
    # then iterdir to check all subpaths
    for subpath in Path(file_path).iterdir():
        # filter the files among the obtained subpaths
        if subpath.is_file():
            # filter the images among these files
            if subpath.suffix in valid_image_extensions:
                image_names.append(subpath)

    return image_names

def get_all_images_rgb(file_path:str):
    images_rgb = []
    
    for subpath in Path(file_path).iterdir():
        if subpath.is_file():
            if subpath.suffix in valid_image_extensions:
                subpath_str = subpath.__str__()
                
                image = cv2.imread(subpath_str)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images_rgb.append(image)

    return images_rgb
    
#---

def normalize(image:list, mask:list):
    # divides the RGB values of the image by 255
    # changes the range of values in the tensor from 0-255 into 0-1
    image = tf.cast(image, tf.float32) / 255.0

    return image, mask


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
