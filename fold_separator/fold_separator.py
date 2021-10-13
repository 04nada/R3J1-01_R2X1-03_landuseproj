import numpy as np
import random

import sys
from PIL import Image
from pathlib import Path
import shutil

# import parent folder
sys.path.append(str(Path.cwd().parent))
import model_functions as model_funcs
import main_params as mp

# --- -----

def set_seed(seed:int):
    np.random.seed(seed)
    random.seed(seed)

def create_separate_dataset_groups(number_of_groups,
*, dataset_dirpath:str, label_names:list, indices:list, output_dirpath:str=None):

    print('== Create Separate Dataset Groups - start ==')

    dataset_regenerator = model_funcs.ReGenerator(
        model_funcs.dataset_generator,
        (dataset_dirpath, label_names)
    )

    dataset = dataset_regenerator.gen()

    # ---

    # define default output directory based on dataset folder name,
    #     if output directory is not explicitly given
    if output_dirpath == None:
        output_dirpath = dataset_dirpath + '_GROUPS'
    
    output_dir_path = Path(output_dirpath)
    
    if output_dir_path.exists():
        shutil.rmtree(output_dir_path)
    output_dir_path.mkdir()

    for g in range(number_of_groups):
        current_group_folder = 'group' + str(g+1)
        current_group_dir_path = output_dir_path / current_group_folder
        current_group_dir_path.mkdir()

        for label in label_names:
            group_class_dir_path = current_group_dir_path / label
            group_class_dir_path.mkdir()
    
    # ---

    for d,datapoint in enumerate(dataset):
        image = datapoint[0]
        label_number = datapoint[1]
        label_name = label_names[label_number]
        
        # ---

        for i,index_list in enumerate(indices):
            if d in index_list:
                group_number = i+1
                group_name = 'group' + str(group_number)
                break

        # ---

        image_file_path = output_dir_path / group_name / label_name / Path(str(d)+'.png')
        
        image_obj = Image.fromarray(np.array(image).astype(np.uint8))
        image_obj.save(image_file_path, format='png')

    print('== Create Separate Dataset Groups - end ==')
