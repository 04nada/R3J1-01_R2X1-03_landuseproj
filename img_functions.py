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

from pathlib import Path
import cv2

valid_image_extensions = [
    '.jpg',
    '.jpeg',
    '.png',
]

# create (label_no, image_tensor) tuples for each individual image
def create_datapoints_from_directory(dataset_path:str, label_names:list):
    print('=== Create Dataset: from Directory - start ===')
    
    datapoints = []
    
    dataset_path_obj = Path(dataset_path)
    dataclasses_paths = [i for i in dataset_path_obj.iterdir() if i.is_dir()]

    # iterdir to check all subpaths
    for i,subpath in enumerate(dataclasses_paths):
        dataclass_path_obj = Path(subpath)

        print('--- CD_D: ' + dataclass_path_obj.name + ' - ' + str(i) + ' of ' + str(len(dataclasses_paths)) + ' classes ---')

        for image in get_all_images_rgb(dataclass_path_obj.__str__()):
            image_class_name = dataclass_path_obj.name
        
            datapoints.append((
                label_names.index(image_class_name),
                image
            ))

    print('--- CD_D: ' + str(len(dataclasses_paths)) + ' of ' + str(len(dataclasses_paths)) + ' classes ---')

    print('=== Create Dataset: from Directory - finish ===')

    return datapoints

### create list of all images inside a folder path + its subfolders
def get_all_images_rgb(file_path:str):
    file_path_obj = Path(file_path)
    images_rgb = []
    
    for subpath in file_path_obj.iterdir():
        if subpath.is_file():
            if subpath.suffix in valid_image_extensions:
                subpath_str = subpath.__str__()
                
                image = cv2.imread(subpath_str)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images_rgb.append(image)

    return images_rgb
    
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
