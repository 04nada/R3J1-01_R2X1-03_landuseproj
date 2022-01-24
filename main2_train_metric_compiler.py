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
    train_confusion_matrices = pickle.load(open(
        str(Path(mp.TRAINING_HISTORIES_DIRECTORY)
            / ('model__fold' + str(f).zfill(2)
                + '__train_confusion_matrices.obj')
        ), 'rb')
    )

    val_confusion_matrices = pickle.load(open(
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
        'val_accuracy': [],
        
        'precision': [],
        'val_precision': [],
        
        'recall': [],
        'val_recall': [],
        
        'F1score': [],
        'val_F1score': []
    }

    for train_confusion_matrix, val_confusion_matrix in zip(train_confusion_matrices, val_confusion_matrices):
        metric_dict['accuracy'].append(
            model_funcs.get_macro_accuracy1(train_confusion_matrix)
        )
        metric_dict['val_accuracy'].append(
            model_funcs.get_macro_accuracy1(val_confusion_matrix)
        )
        
        metric_dict['precision'].append(
            model_funcs.get_macro_precision(train_confusion_matrix)
        )
        metric_dict['val_precision'].append(
            model_funcs.get_macro_precision(val_confusion_matrix)
        )
        
        metric_dict['recall'].append(
            model_funcs.get_macro_recall(train_confusion_matrix)
        )
        metric_dict['val_recall'].append(
            model_funcs.get_macro_recall(val_confusion_matrix)
        )
        
        metric_dict['F1score'].append(
            model_funcs.get_macro_F1(train_confusion_matrix)
        )
        metric_dict['val_F1score'].append(
            model_funcs.get_macro_F1(val_confusion_matrix)
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
        val_accuracy = metric_dict['val_accuracy'],
        val_precision = metric_dict['val_precision'],
        val_recall = metric_dict['val_recall']
    )
    
