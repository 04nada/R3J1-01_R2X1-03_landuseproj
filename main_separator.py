from fold_separator import fold_separator

import main_params as mp
import model_functions as model_funcs

fold_separator.set_seed(mp.SEED)

custom_output_dirpath = str(mp.ROOT_DIRECTORY / 'datasets' / 'train_images' / 'trueclass_240x240_sortbyclass_actual_folds' / 'groups')

fold_separator.create_separate_dataset_groups(
    mp.FOLDS,
    dataset_dirpath = str(mp.TRAIN_DATASET_DIRECTORY),
    label_names = mp.label_names,
    indices = model_funcs.random_index_partition(
        mp.FOLDS,
        model_funcs.get_dataset_size(str(mp.TRAIN_DATASET_DIRECTORY))
    ),
    output_dirpath = custom_output_dirpath
)
