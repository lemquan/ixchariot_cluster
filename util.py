import time
import pickle
import numpy as np
import sys
from folds import Fold

if sys.platform == 'darwin':
    import matplotlib as mil
    mil.use('TkAgg')

    import matplotlib.pyplot as plt
    import seaborn as sns
    plot_on = True
    print "Running OS X"

elif sys.platform == 'linux' or sys.platform == 'linux2':
    print "Running Linux. Plots are saved."
    import matplotlib as mil
    mil.use('Agg')
    import matplotlib.pyplot as plt
    plot_on = False

def timeit(func):
    def timed(*args, **kwargs):
        ts = time.time()
        res = func(*args, **kwargs)
        te = time.time()

        print '%r %f sec' % \
                    (func.__name__, te-ts)
        return res
    return timed


def create_cv_folds(X, num_folds=5):
    ''' Create the cross validation folds. Time series so we create an additional fold for chaining '''

    folds = []
    # split the train-test set 80/20 keeping the time order
    sp = int(X.shape[0] * 0.80)

    train = X[:sp, :]
    test = X[sp:, :]

    train_fold = Fold(train[:,1:], train[:,0])
    test_fold = Fold(test[:,1:], test[:,0])

    pickle.dump(train_fold, open('data/train_data.p', 'wb'))
    pickle.dump(test_fold, open('data/test_data.p', 'wb'))

    # create the n-Folds for cross validation
    fs = int(train.shape[0] / (num_folds+1))
    for f in range(num_folds+1):
        data = train[ (fs*f):fs*(f+1), :]
        fold = Fold(data[:,1:], data[:,0])
        folds.append(fold)
    pickle.dump(folds, open('data/cv.p', 'wb'))
    return folds, train_fold, test_fold

def get_cv_folds(folds, num_folds=5):
    train_X = []
    train_y = []

    for i in folds:
        train_X.append(i.get_data())
        train_y.append(i.get_target())

    train_X = np.vstack(train_X).squeeze()
    train_y = np.vstack(train_y).reshape(train_X.shape[0])
    return train_X, train_y


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    target_names = ['Bit Torrent', 'Netflix', 'HTTPs', 'Facebook']

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
