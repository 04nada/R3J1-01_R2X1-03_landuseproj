import main_params as mp
import model_functions as model_funcs

from pathlib import Path
import pickle

folds_epochs = [
    (1, 8),
    (2, 14),
    (3, 7),
    (4, 15),
    (5, 12)
]

for f,epochs in folds_epochs:
    confusion_matrices = pickle.load(open(
        str(Path(mp.TRAINING_HISTORIES_DIRECTORY)
            / ('model__fold' + str(f).zfill(2)
                + '__train_confusion_matrices.obj')
        ), 'rb')
    )

    history = pickle.load(open(
        str(Path(mp.TRAINING_HISTORIES_DIRECTORY)
            / ('model__fold' + str(f).zfill(2)
                + '__history.obj')
        ), 'rb')
    )

    # ---
    
    metric_dict = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'F1score': [] 
    }

    for confusion_matrix in confusion_matrices:
        metric_dict['accuracy'].append(
            model_funcs.get_macro_accuracy1(confusion_matrix)
        )

        metric_dict['precision'].append(
            model_funcs.get_macro_precision(confusion_matrix)
        )

        metric_dict['recall'].append(
            model_funcs.get_macro_recall(confusion_matrix)
        )

        metric_dict['F1score'].append(
            model_funcs.get_macro_F1(confusion_matrix)
        )

    pickle.dump(metric_dict, open(
        str(Path(mp.TRAINING_HISTORIES_DIRECTORY)
            / ('model__fold' + str(f).zfill(2)
                + '__training_metrics.obj')
        ), 'wb')
    )

    model_funcs.plot_training_graphs(
        epochs,
        train_accuracy = metric_dict['accuracy'],
        train_precision = metric_dict['precision'],
        train_recall = metric_dict['recall'],
        train_F1score = metric_dict['F1score']
    )
    
