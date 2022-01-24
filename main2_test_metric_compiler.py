import main_params as mp
import model_functions as model_funcs

from pathlib import Path
import pickle

for f in range(mp.FOLDS):
    test_confusion_matrix = pickle.load(open(
        str(Path(mp.TESTING_RESULTS_DIRECTORY)
            / ('model__fold' + str(f+1).zfill(2)
                + '__test_confusion_matrices.obj')
        ), 'rb')
    )

    results = pickle.load(open(
        str(Path(mp.TRAINING_HISTORIES_DIRECTORY)
            / ('model__fold' + str(f+1).zfill(2)
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

    for train_confusion_matrix, val_confusion_matrix in zip(train_confusion_matrices, val_confusion_matrices):
        metric_dict['accuracy'].append(
            model_funcs.get_macro_accuracy1(test_confusion_matrix)
        )
        
        metric_dict['precision'].append(
            model_funcs.get_macro_precision(test_confusion_matrix)
        )
        
        metric_dict['recall'].append(
            model_funcs.get_macro_recall(test_confusion_matrix)
        )
        
        metric_dict['F1score'].append(
            model_funcs.get_macro_F1(test_confusion_matrix)
        )
        
    pickle.dump(metric_dict, open(
        str(Path(mp.TRAINING_HISTORIES_DIRECTORY)
            / ('model__fold' + str(f+1).zfill(2)
                + '__test_metrics.obj')
        ), 'wb')
    )

    model_funcs.plot_test_graphs(
        epochs,
        model_accuracy = metric_dict['accuracy'],
        model_precision = metric_dict['precision'],
        model_recall = metric_dict['recall'],
        model_F1score = metric_dict['F1score']
    )
