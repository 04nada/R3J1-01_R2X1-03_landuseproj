import tensorflow as tf
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

class Dataset:
    ### Initialization
    
    def __init__(self, *, directory, **kwargs):
        self.__directory = Path(directory)
        self.__subdirectories = []
        
        self.__classes = kwargs.get('classes', None)

        self.__datapoints = []
        
        # ---

        # create subfolders if specific classes are specified
        if self.get_classes():
            for c in self.get_classes():
                class_directory = Path(self.get_directory(), c)
                
                if not class_directory.exists():
                    class_directory.mkdir()

        # get all immediate subdirectories in main directory
        for subpath in Path.iterdir(self.directory):
            if subpath.is_dir():
                self.add_subdirectory(Path(subpath))

        # ---

        # create datapoints for each object in
        for sd in self.get_subdirectories():
            for subpath in Path.iterdir(sd):
                if subpath.is_file():
                    dp = Datapoint(subpath, sd)
                    self.add_datapoint(dp)

    #---
                    
    ### Getters

    def get_directory(self):
        return self.__directory

    def get_subdirectories(self):
        return self.__subdirectories

    def get_classes(self):
        return self.__classes

    def get_datapoints(self):
        return self.__datapoints

    #---

    ### Setters

    def add_subdirectory(self, sd):
        self.__subdirectories.append(sd)

    def add_datapoint(self, dp):
        self.__datapoints.append(dp)
        
class Datapoint:
    ### Initialization
    
    def __init__(self, abspath:str, label:str):
        ### Image
        
        self.__image_abspath = Path(abspath)

        self.image_filename = self.get_image_path().stem
        self.image_directory = os.path.dirname(self.image_path)
        self.__image_extension = self.get_image_path().suffix
        self.__image_fileformat = self.__image_extension[1:]

        # ---

        self.__image_nptensor = cv2.imread(self.image_root)
        self.__image_nptensor = cv2.cvtColor(self.image_tensor, cv2.COLOR_BGR2RGB)

        # ---

        self.__label_name = label
        
    # ---

    ### Getters

    def get_image_path(self):
        return self.__image_abspath
    
    def get_image_tensor(self):
        return self.__image_nptensor

    def get_label_name(self):
        return self.__label_name
