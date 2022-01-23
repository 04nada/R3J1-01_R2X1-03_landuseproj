import main_params as mp

from pathlib import Path
import pickle

folds_epochs = [
    (1, 8),
    (2, 14),
    (3, 7),
    (4, 15),
    (5, 12)
]

# ---

for f,epochs in folds_epochs:
    matrices = []
    
    for e in range(epochs):
        confusion_matrix = pickle.load(open(
            str(Path(mp.TRAINING_HISTORIES_DIRECTORY)
                / ('model__fold' + str(f).zfill(2)
                    + '__epoch' + str(e+1).zfill(2)
                    + '__train_confusion_matrix.obj')
            ), 'rb')
        )

        matrices.append(confusion_matrix)

    pickle.dump(matrices, open(
        str(Path(mp.TRAINING_HISTORIES_DIRECTORY)
            / ('model__fold' + str(f).zfill(2)
               + '__train_confusion_matrices.obj')
        ), 'wb')
    )
