import numpy as np

from PIL import Image
import os
import cv2
from pathlib import Path
import shutil

from image_gridifier import image_gridifier

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

class LargeDatapoint:
    def __init__(self, imgpath:str):
        self.image_path = imgpath

        # get filename (no extension) and directory name of image from path
        self.image_filename = Path(self.image_path).stem
        self.image_directory = os.path.dirname(self.image_path)
        self.image_extension = Path(self.image_path).suffix
        
        # ---
        
        # load src_image as numpy array
        self.image_nptensor = cv2.imread(self.image_path)
        self.image_nptensor = cv2.cvtColor(self.image_nptensor, cv2.COLOR_BGR2RGB)

    def init_labels(self, lblpath:str, lbllookup_rgb2i:dict, lbllookup_i2n:list):
        self.labels_path = lblpath

        self.labels_filename = Path(self.labels_path).stem
        self.labels_directory = os.path.dirname(self.labels_path)
        self.labels_extension = ".png"
        self.labels_fileformat = "png"

        self.label_lookup_rgb_to_index = lbllookup_rgb2i
        self.label_lookup_index_to_name = lbllookup_i2n

        # ---

        self.labels_nptensor = cv2.imread(self.image_path)
        self.labels_nptensor = cv2.cvtColor(self.image_nptensor, cv2.COLOR_BGR2RGB)

    def set_datapoints(self, imgdir_sortbyclass:str, grid_HEIGHT:int, grid_WIDTH:int, is_last_row_bigger=False, is_last_column_bigger=False):
        gridified_image_tensor = image_gridifier.gridify_image(self.image_nptensor, grid_HEIGHT, grid_WIDTH, is_last_row_bigger, is_last_column_bigger)
        gridified_label_tensor = image_gridifier.gridify_image(self.labels_nptensor, grid_HEIGHT, grid_WIDTH, is_last_row_bigger, is_last_column_bigger)

        self.datapoints = []

        for gr,grid_row in enumerate(gridified_image_tensor):
            for subimg,subimage in enumerate(grid_row):
                filename = (self.image_filename
                    + ' '
                    + chr(65+subimg)    # column letter, A-?
                    + str(gr+1)         # row number, 1-?
                )

                # initialize a datapoint class
                current_datapoint = Datapoint()
                current_datapoint.init_label(
                    gridified_label_tensor[gr][subimg],
                    self.label_lookup_rgb_to_index,
                    self.label_lookup_index_to_name
                )

                # identify fileroot for datapoint based on its classification as a subfolder
                fileroot = os.path.join(
                    imgdir_sortbyclass,
                    current_datapoint.get_label_name(),
                    filename + ".png"
                )
                
                current_datapoint.init_path(fileroot)

    def create_datapoints(self):
        for dp in self.datapoints:
            dp.create_file()
    
class Datapoint:
    ### Initializers
    
    def __init__(self):
        pass
    
    def init_path(self, imgroot:str):
        ### Image
        
        self.image_fileroot = imgroot

        self.image_filename = Path(self.image_path).stem
        self.image_directory = os.path.dirname(self.image_path)
        self.image_extension = Path(self.image_path).suffix
        self.image_fileformat = image_extension[1:]

        # ---

        self.image_nptensor = cv2.imread(self.image_fileroot)
        self.image_nptensor = cv2.cvtColor(self.image_tensor, cv2.COLOR_BGR2RGB)
    
    def init_label(self, lblcolor:list, lbllookup_rgb2i:dict, lbllookup_i2n:list):      
        self.label_color = lblcolor
        self.label_number = self.label_lookup_rgb_to_index.get(self.label_color)
        self.label_name = self.label_lookup_index_to_name[self.label_number]

    # ---

    ### Getters
    
    def get_image_tensor(self):
        return self.image_tensor

    def get_label_number(self):
        return self.label_number

    def get_label_name(self):
        return self.label_name

    # ---

    def create_file(self):
        if not os.path.exists(self.image_directory):
            os.makedirs(self.image_directory)

        img_array = np.array(self.image_tensor).astype(np.uint8)
        img_array.save(self.image_fileroot + self.image_extension, format=self.image_fileformat)
