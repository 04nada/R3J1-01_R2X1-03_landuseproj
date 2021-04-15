import numpy as np

from PIL import Image
import os
import cv2
from pathlib import Path
import shutil

# ---

# sample matrices

small_sample = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]

large_sample = []
for i in range(1000):
    large_sample.append([i*1000+j for j in range(1000)])

# ---

# creates a folder containing subimages that would have been generates
#       from gridify_image() with the same parameters
# this function accepts image path, instead of an image matrix
def generate_grid_image_files(src_image_path:str, grid_HEIGHT:int, grid_WIDTH:int, is_last_row_bigger=False, is_last_column_bigger=False):
    # get filename (no extension) and directory name of src_image from path
    src_image_filename = Path(src_image_path).stem
    src_image_directory = os.path.dirname(src_image_path)

    # create empty folder with same name as src_image, on the same level
    subimage_directory = os.path.join(src_image_directory, src_image_filename)
    
    if os.path.exists(subimage_dirname):
        # if folder already exists, delete it
        shutil.rmtree(subimage_directory)
        
    os.mkdir(subimage_dirname)

    # ---

    # load src_image as numpy array
    src_image = cv2.imread(src_image_path)
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)

    # convert src_image from numpy array to list
    src_image = np.array(src_image).tolist()
    
    gridified_image = gridify_image(src_image, grid_HEIGHT, grid_WIDTH, is_last_row_bigger, is_last_column_bigger)
    
    for gr,grid_row in enumerate(gridified_image):
        for subimg,subimage in enumerate(grid_row):
            # append LETTERnumber code for each subimage
            filename = (src_image_filename
                + ' '
                + chr(65+subimg)    # column letter, A-?
                + str(gr+1)         # row number, 1-?
            )

            fileroot = os.path.join(subimage_directory, filename)
            
            # save subimage in previously made folder
            current_image = Image.fromarray(np.array(subimage).astype(np.uint8))
            current_image.save(fileroot+'.png', format='png')

# generate matrix of subimage matrices from a larger image matrix
# divides the larger image into grid by given height and width
def gridify_image(image:list, grid_HEIGHT:int, grid_WIDTH:int, is_last_row_bigger=False, is_last_column_bigger=False):
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
    for gr in range(NUM_ROWS):
        grid_row = []

        for gc in range(NUM_COLUMNS):
            subimage = []

            # compute subimage height based on row number and
            #       if the last row is bigger or smaller than the rest
            if gr < NUM_ROWS-1:
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

            for ir in range(subimage_HEIGHT):
                subimage_row = []

                # compute subimage width based on column number and
                #       if the last row is bigger or smaller than the rest
                if gc < NUM_COLUMNS-1:
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
                        
                for ic in range(subimage_WIDTH):
                    subimage_pixel = image[gr*grid_HEIGHT + ir][gc*grid_WIDTH + ic]
                    
                    subimage_row.append(subimage_pixel)

                subimage.append(subimage_row)
                
            grid_row.append(subimage)

        gridified_image.append(grid_row)

    return gridified_image
