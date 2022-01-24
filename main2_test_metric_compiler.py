import main_params as mp
import model_functions as model_funcs

from pathlib import Path
import pickle

fold = 2
epoch = 12

test_confusion_matrix = pickle.load(open(
    str(Path(mp.TESTING_RESULTS_DIRECTORY)
        / ('model__fold' + str(fold).zfill(2)
            + '__epoch' + str(epoch).zfill(2)
            + '__test_confusion_matrix.obj')
    ), 'rb')
)

results = pickle.load(open(
    str(Path(mp.TESTING_RESULTS_DIRECTORY)
        / ('model__fold' + str(fold).zfill(2)
            + '__epoch' + str(epoch).zfill(2)
            + '__results.obj')
    ), 'rb')
)

# ---

metric_dict = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'F1score': []
}

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
    str(Path(mp.TESTING_RESULTS_DIRECTORY)
        / ('model__fold' + str(fold).zfill(2)
            + '__epoch' + str(epoch).zfill(2)
            + '__test_metrics.obj')
    ), 'wb')
)

model_funcs.plot_test_graphs(
    1,
    test_accuracy = metric_dict['accuracy'],
    test_precision = metric_dict['precision'],
    test_recall = metric_dict['recall'],
    test_F1score = metric_dict['F1score']
)
