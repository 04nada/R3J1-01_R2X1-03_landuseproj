from pathlib import Path

from fold_separator import fold_separator

import main_params as mp
import model_functions as model_funcs

fold_separator.set_seed(mp.SEED)

#preset_dataset_dirpath = str(Path(mp.ROOT_DIRECTORY) / 'datasets' / 'train_images' / 'trueclass_240x240_sortbyclass_actual')
preset_dataset_dirpath = str(Path(mp.ROOT_DIRECTORY) / 'datasets' / 'train_images' / 'trueclass_240x240_sortbyclass_actual2')

#custom_output_dirpath = str(Path(mp.ROOT_DIRECTORY) / 'datasets' / 'train_images' / 'trueclass_240x240_sortbyclass_actual_folds' / 'groups')
custom_output_dirpath = str(Path(mp.ROOT_DIRECTORY) / 'datasets' / 'train_images' / 'trueclass_240x240_sortbyclass_actual2_folds' / 'groups')

fold_separator.create_separate_dataset_groups(
    mp.FOLDS,
    dataset_dirpath = preset_dataset_dirpath,
    label_names = mp.label_names,
    indices = model_funcs.random_index_partition(
        mp.FOLDS,
        model_funcs.get_dataset_size(preset_dataset_dirpath)
    ),
    output_dirpath = custom_output_dirpath
)
