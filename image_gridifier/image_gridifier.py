import numpy as np
import math

from PIL import Image
from pathlib import Path
import shutil

sys.path.append(str(Path.cwd().parent))
import model_functions as model_funcs

# --- -----

### Sorted Image Gridifier for all images and labels in their respective directories

def generate_sorted_grid_image_files_by_directory(start_index=0, end_index=math.inf,
*, image_folder_dirpath:str, colormap_folder_dirpath:str,
subimage_height:int, subimage_width:int, lookup_colors:dict, label_names:int):
    print('=== Sorted Image Gridifier: by Directory - start ===\n')
    
    image_folder_dir_path = Path(image_folder_dirpath)
    colormap_folder_dir_path = Path(colormap_folder_dirpath)

    image_file_paths = []
    colormap_file_paths = []
        
    # search across all image files
    for image_file_path in image_folder_dir_path.iterdir():
        if image_file_path.is_file() and image_file_path.suffix in model_funcs.VALID_IMAGE_EXTENSIONS:
            # store all image filepaths from each name folder
            images_folder_filepaths.append(image_filepath)

    # search across all name folders             
    for colormap_file_path in colormap_folder_dir_path.iterdir():
        if colormap_file_path.is_file() and colormap_file_path.suffix in model_funcs.VALID_IMAGE_EXTENSIONS:
            # store all image filepaths from each name folder
            colormap_file_paths.append(labels_filepath)

    # run Sorted Image Gridifier for each imagepath-labelspath pair in the specified directories                      
    for i, (image_file_path, colormap_file_path) in enumerate(zip(image_file_paths, colormap_file_paths)):
        if i < start_index:
            continue

        if i >= end_index:
            continue

        print('--- SIG_D: ' + str(i) + ' of ' + str(len(images_folder_filepaths)) + ' images completed ---\n')

        generate_sorted_grid_image_files(
            subimage_height = subimage_height,
            subimage_width = subimage_width,
            image_abspath = str(image_file_path),
            labels_abspath = str(colormap_file_path),
            lookup_colors = lookup_colors,
            label_names = label_names
        )

        print('')
        
    print('--- SIG_D: Image ' + str(len(image_file_paths)) + ' of ' + str(len(image_file_paths)) + '---\n')
        
    print('=== Sorted Image Gridifier: by Directory - finish ===')


### Generate Subimages from gridify_image(), unsorted all into a folder

def generate_unsorted_grid_image_files(is_last_row_bigger=False, is_last_column_bigger=False,
*, subimage_height:int, subimage_width:int, image_filepath:str):
    # creates a folder containing subimages that would have been generated
    #       from gridify_image() with the same parameters
    # this function accepts <image_abspath>, instead of an image matrix

    print('== Unsorted Image Gridifier - start ==')

    # ---

    ### Instantiate the source image variables

    image_dir_path = Path(image_filepath).parent
    image_filename = Path(image_filepath).stem

    # create empty folder with same name as src_image, on the same level
    subimage_directory = image_dir_path / image_filename

    print('- for: ' + src_image_filename)
    # if folder already exists, delete it
    if subimage_directory.exists():
        shutil.rmtree(subimage_directory)
        
    subimage_directory.mkdir()

    # ---

    ### Process and gridify the source image

    src_image = generate_rgb_image_from_path(image_filepath)
    
    gridified_image = gridify_image(
        src_image,
        grid_height, grid_width,
        is_last_row_bigger, is_last_column_bigger
    )

    # ---

    ### Save all subimages in gridified_image as image files
    
    for gi_r,gridified_image_row in enumerate(gridified_image):
        print('-- UIG: Row ' + str(gi_r) + ' of ' + str(len(gridified_image)) + ' --')

        for gi_c,subimage in enumerate(gridified_image_row):
            # append LETTERnumber code for each subimage
            filename = (src_image_filename
                + ' '
                + chr(65+gi_c)    # column letter, A-?
                + str(gi_r+1)         # row number, 1-?
            )

            file_abspath = subimage_directory / filename
            
            # save subimage in previously made folder
            current_image = Image.fromarray(np.array(subimage).astype(np.uint8))
            current_image.save(file_abspath.with_suffix('.png'), format='png')

    print('-- UIG: Row ' + str(len(gridified_image)) + ' of ' + str(len(gridified_image)) + ' --')

    print('== Unsorted Image Gridifier - finish ==')

# ---

### Generate Subimages from gridify_image(), sorted by class/label into a folder
  
def generate_sorted_grid_image_files(is_last_row_bigger=False, is_last_column_bigger=False,
*, subimage_height:int, subimage_width:int, image_filepath:str, colormap_filepath:str, lookup_colors:dict, label_names:list):
    # also uses the <labels_abspath>, <lookup_colors>, and <label_names> keywords
    #       to indicate the image with the corresponding label colors
    
    print('== Sorted Image Gridifier - start ==')

    # ---

    ### Instantiate the source image variables

    image_dir_path = Path(image_filepath).parent
    image_filename = Path(image_filepath).stem

    # create empty folder with same name as src_image, on the same level
    subimage_dir_path = image_dir_path / image_filename

    # ---

    ### Create image 

    print('- for: ' + src_image_filename)

    # if folder already exists, delete it, to force empty folder
    if subimage_dir_path.exists():
        shutil.rmtree(subimage_dir_path)
        
    subimage_dir_path.mkdir()

    # create empty subfolder for each class/label
    for label in label_names:
        class_directory = subimage_dir_path / label

        class_directory.mkdir()
    
    # ---

    ### Process and gridify the source image
    
    src_image = generate_rgb_image_from_path(image_filepath)

    gridified_image = gridify_image(
        src_image,
        subimage_height, subimage_width,
        is_last_row_bigger, is_last_column_bigger
    )


    ### Process the colormap image
    
    labels_image = generate_rgb_image_from_path(colormap_filepath)
    
    # ---
    
    for gi_r,gridified_image_row in enumerate(gridified_image):
        print('-- SIG: Row ' + str(gi_r) + ' of ' + str(len(gridified_image)) + ' --')
        
        for gi_c,subimage in enumerate(gridified_image_row):
            # identify the class name given the class color
            class_color = tuple(labels_image[gi_r][gi_c])
            class_number = lookup_colors.get(class_color)
            class_name = label_names[class_number]
            
            # append LETTERnumber code for each subimage
            subimage_filename = (src_image_filename
                + ' '
                + chr(65+gi_c)          # column letter, A-?
                + str(gi_r+1)           # row number, 1-?
            )

            subimage_file_path = (subimage_dir_path / class_name / filename).with_suffix('.png')
            
            # save subimage in previously made folder
            subimage_obj = Image.fromarray(np.array(subimage).astype(np.uint8))
            subimage_obj.save(subimage_file_path, format='png')

    print('-- SIG: Row ' + str(len(gridified_image)) + ' of ' + str(len(gridified_image)) + ' --')
        
    print('== Sorted Image Gridifier - finish ==')

# --- -----

# Gridify Image, into a matrix of matrices

def gridify_image(image:list, subimage_height:int, subimage_width:int, is_last_row_bigger=False, is_last_column_bigger=False):
    # divides the larger image into grid by given height and width
    
    # last two boolean parameters determine if last row/column will be bigger or smaller,
    #       if the grid lengths do not divide the image lengths evenly

    image_height = len(image)
    image_width = len(image[0])

    if is_last_row_bigger:
        # floor division
        NUM_ROWS = image_height // subimage_height
    else:
        # ceiling division
        NUM_ROWS = -(-image_height // subimage_height)

    if is_last_column_bigger:
        # floor division
        NUM_COLUMNS = image_width // subimage_width
    else:
        # ceiling division
        NUM_COLUMNS = -(-image_width // subimage_width)

    # ---

    gridified_image = []

    print("GRID COLUMNS: " + str(NUM_COLUMNS))
    print("GRID ROWS: " + str(NUM_ROWS))

    # O(n^4) nested lists end me now
    for gi_r in range(NUM_ROWS):
        gridified_image_row = []
        
        # compute subimage height based on row number and
        #       from if the last row is bigger or smaller than the rest
        if gi_r < NUM_ROWS-1:
            # subimage uses standard grid height for every row except the last
            current_subimage_height = subimage_height
        else:
            if image_height % subimage_height == 0:
                # if the grid height evenly divides image height
                #       then use that standard grid height for the last row anyway
                current_subimage_height = subimage_height
            else:
                if is_last_row_bigger:
                    current_subimage_height = (image_height % subimage_height) + subimage_height
                else:
                    current_subimage_height = image_height % subimage_height

        # ---
        
        for gi_c in range(NUM_COLUMNS):
            subimage = []

            # compute subimage width based on column number and
            #       from if the last row is bigger or smaller than the rest
            if gi_c < NUM_COLUMNS-1:
                # current subimage uses standard subimage width
                #       for every column except the last
                current_subimage_width = subimage_width
            else:
                if image_width % subimage_width == 0:
                    # if the subimage width evenly divides image width
                    #       then use that standard grid width for the last column anyway
                    current_subimage_width = subimage_width
                else:
                    if is_last_column_bigger:
                        current_subimage_width = (image_width % subimage_width) + subimage_width
                    else:
                        current_subimage_width = image_width % subimage_width
            
            # ---
            
            for si_r in range(subimage_height):
                subimage_row = []
  
                for si_c in range(subimage_width):
                    subimage_pixel = image[gi_r*subimage_height + si_r][gi_c*subimage_width + si_c]
                    subimage_row.append(subimage_pixel)

                subimage.append(subimage_row)
                
            gridified_image_row.append(subimage)

        gridified_image.append(gridified_image_row)

    return gridified_image
