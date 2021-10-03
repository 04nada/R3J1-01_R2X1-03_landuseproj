import numpy as np
import random

import sys
from pathlib import Path

sys.path.append(Path.cwd().parent)
import model_functions as model_funcs

def set_seed(seed:int):
    np.random.seed(seed)
    random.seed(seed)

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

def funcy(
*, dataset_dirpath:str, label_names:list, dataset:'generator'):
    dataset_dir_path = Path(dataset_dirpath)

    class_dir_paths = (i for i in dataset_Path.iterdir() if i.is_dir())

    

    


