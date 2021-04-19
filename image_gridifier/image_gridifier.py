import numpy as np

from PIL import Image
import cv2
from pathlib import Path
import shutil

# ---

lookup_rgb_to_index = {
    (97,64,31): 0,	    # agricultural - brown - #61401F
    (160,32,239): 1,	    # commercial - purple - #A020EF
    (221,190,170): 2,       # industrial - beige - #DDBEAA
    (238,0,2): 3,   	    # institutional - red - #ED0000
    (45,137,86): 4,	    # recreational - green - #2D8956
    (254,165,0): 5,	    # residential - yellow - #FEA500
    (0,0,88): 6		    # transport - dark blue - #000057
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

# ---

def generate_grid_image_files(grid_HEIGHT:int, grid_WIDTH:int, is_last_row_bigger=False, is_last_column_bigger=False,
*, image_abspath:str):
    # creates a folder containing subimages that would have been generated
    #       from gridify_image() with the same parameters
    # this function accepts <image_abspath>, instead of an image matrix

    print('=== Unsorted Image Gridifier - start ===')

    # ---

    ### Instantiate the source image variables

    src_image_abspath = image_abspath
    src_image_directory = Path(src_image_abspath).parent
    src_image_filename = Path(src_image_abspath).stem

    # create empty folder with same name as src_image, on the same level
    subimage_directory = src_image_directory / src_image_filename

    print('- for: ' + src_image_filename)
    if subimage_directory.exists():
        # if folder already exists, delete it
        shutil.rmtree(subimage_directory)
        
    subimage_directory.mkdir()

    # ---

    ### Gridify the source image

    # load src_image as numpy array
    src_image = cv2.imread(src_image_abspath)
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    
    gridified_image = gridify_image(src_image, grid_HEIGHT, grid_WIDTH, is_last_row_bigger, is_last_column_bigger)

    # ---
    
    for gr,grid_row in enumerate(gridified_image):
        for subimg,subimage in enumerate(grid_row):
            # append LETTERnumber code for each subimage
            filename = (src_image_filename
                + ' '
                + chr(65+subimg)    # column letter, A-?
                + str(gr+1)         # row number, 1-?
            )

            file_abspath = subimage_directory / filename
            
            # save subimage in previously made folder
            current_image = Image.fromarray(np.array(subimage).astype(np.uint8))
            current_image.save(file_abspath.with_suffix('.png'), format='png')

        print('-- Finished: Row ' + str(gr+1) + ' of ' + str(len(gridified_image)) + ' --')

    print('=== Unsorted Image Gridifier - finish ===')

  
def generate_sorted_grid_image_files(grid_HEIGHT:int, grid_WIDTH:int, is_last_row_bigger=False, is_last_column_bigger=False,
*, image_abspath:str, labels_abspath:str, lookup_colors:dict, label_names:list):
    # also uses the <labels_abspath>, <lookup_colors>, and <label_names> keyowrds
    #       to indicate the image with the corresponding label colors
    
    print('=== Sorted Image Gridifier - start ===')

    # ---

    ### Instantiate the source image variables

    src_image_abspath = image_abspath
    src_image_directory = Path(src_image_abspath).parent
    src_image_filename = Path(src_image_abspath).stem

    # create empty folder with same name as src_image, on the same level
    subimage_directory = src_image_directory / src_image_filename

    print('- for: ' + src_image_filename)
    if subimage_directory.exists():
        # if folder already exists, delete it, to force empty folder
        shutil.rmtree(subimage_directory)
        
    subimage_directory.mkdir()

    # create empty subfolder for each class/label
    for label in label_names:
        class_directory = subimage_directory / label

        if not class_directory.exists():
            class_directory.mkdir()
    
    # ---

    ### Process and gridify the source image
    
    src_image = cv2.imread(src_image_abspath)
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)

    gridified_image = gridify_image(src_image, grid_HEIGHT, grid_WIDTH, is_last_row_bigger, is_last_column_bigger)


    ### Process the labels image
    
    labels_image = cv2.imread(labels_abspath)
    labels_image = cv2.cvtColor(labels_image, cv2.COLOR_BGR2RGB)
    
    # ---
    
    for gr,grid_row in enumerate(gridified_image):
        for subimg,subimage in enumerate(grid_row):
            # identify the class name given the class color
            class_color = tuple(labels_image[gr][subimg])
            class_number = lookup_colors.get(class_color)
            class_name = label_names[class_number]
            
            # append LETTERnumber code for each subimage
            filename = (src_image_filename
                + ' '
                + chr(65+subimg)    # column letter, A-?
                + str(gr+1)         # row number, 1-?
            )

            file_abspath = subimage_directory / class_name / filename
            
            # save subimage in previously made folder
            current_image = Image.fromarray(np.array(subimage).astype(np.uint8))
            current_image.save(file_abspath.with_suffix('.png'), format='png')

        print('-- Finished: Row ' + str(gr+1) + ' of ' + str(len(gridified_image)) + ' --')

    print('=== Sorted Image Gridifier - finish ===')


def gridify_image(image:list, grid_HEIGHT:int, grid_WIDTH:int, is_last_row_bigger=False, is_last_column_bigger=False):
    # generate matrix of subimage matrices from a larger image matrix
    # divides the larger image into grid by given height and width
    
    # last two parameters determine if last row/column will be bigger or smaller,
    #       if the grid lengths do not divide the image lengths evenly

    image_HEIGHT = len(image)
    image_WIDTH = len(image[0])

    if is_last_row_bigger:
        # floor division
        NUM_ROWS = len(image)//grid_HEIGHT
    else:
        # ceiling division
        NUM_ROWS = -(-len(image)//grid_HEIGHT)

    if is_last_column_bigger:
        # floor division
        NUM_COLUMNS = len(image)//grid_WIDTH
    else:
        # ceiling division
        NUM_COLUMNS = -(-len(image)//grid_WIDTH)

    # ---

    gridified_image = []

    print("GRID COLUMNS: " + str(NUM_COLUMNS))
    print("GRID ROWS: " + str(NUM_ROWS))

    # O(n^4) nested lists end me now
    for g_r in range(NUM_ROWS):
        grid_row = []
        
        # compute subimage height based on row number and
        #       from if the last row is bigger or smaller than the rest
        if g_r < NUM_ROWS-1:
            # subimage uses standard grid height for every row except the last
            subimage_HEIGHT = grid_HEIGHT
        else:
            if image_HEIGHT % grid_HEIGHT == 0:
                # if the grid height evenly divides image height
                #       then use that standard grid height for the last row anyway
                subimage_HEIGHT = grid_HEIGHT
            else:
                if is_last_row_bigger:
                    subimage_HEIGHT = (image_HEIGHT % grid_HEIGHT) + grid_HEIGHT
                else:
                    subimage_HEIGHT = image_HEIGHT % grid_HEIGHT

        # ---
        
        for g_c in range(NUM_COLUMNS):
            subimage = []

            # compute subimage width based on column number and
            #       from if the last row is bigger or smaller than the rest
            if g_c < NUM_COLUMNS-1:
                # subimage uses standard grid width for every column except the last
                subimage_WIDTH = grid_WIDTH
            else:
                if image_WIDTH % grid_WIDTH == 0:
                    # if the grid width evenly divides image width
                    #       then use that standard grid width for the last column anyway
                    subimage_WIDTH = grid_WIDTH
                else:
                    if is_last_column_bigger:
                        subimage_WIDTH = (image_WIDTH % grid_WIDTH) + grid_WIDTH
                    else:
                        subimage_WIDTH = image_WIDTH % grid_WIDTH
            
            # ---
            
            for si_r in range(subimage_HEIGHT):
                subimage_row = []
  
                for si_c in range(subimage_WIDTH):
                    subimage_pixel = image[g_r*grid_HEIGHT + si_r][g_c*grid_WIDTH + si_c]
                    
                    subimage_row.append(subimage_pixel)

                subimage.append(subimage_row)
                
            grid_row.append(subimage)

        gridified_image.append(grid_row)

    return gridified_image
