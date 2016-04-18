import pandas as pd
import numpy as np
import sys
import pickle
from util import timeit
from folds import Fold
from sklearn import mixture, cluster, metrics

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


@timeit
def train_model(train, test):
    n_comps = len(np.unique(train.get_target()))
    #igmm = mixture.GMM(n_components=n_comps, n_iter= 3000) # 35%
    igmm = mixture.DPGMM(n_components=n_comps, n_iter=3000)

    igmm.fit(train.get_data())
    y_train_preds = igmm.predict(train.get_data())
    y_test_preds = igmm.predict(test.get_data())

    train_err = metrics.f1_score(train.get_target(), y_train_preds)
    print 'accuracy for train:', train_err      # 37.8%

    test_err = metrics.f1_score(test.get_target(), y_test_preds)
    print 'accuracy for test:', test_err        # 32.2%

    pred_train = Fold(train.get_data(), y_train_preds)
    pred_test = Fold(test.get_data(), y_test_preds)

    #plot_clusters(pred_train, 'pred_train')
    #plot_clusters(pred_test, 'pred_test')


def plot_clusters(data, s='train'):
    f, ax = plt.subplots(1,1, figsize=(15,12))
    labels = ['Bit Torrent', 'Netflix', 'HTTP', 'FB']
    for i, c in enumerate('krgb'):
        print '--- cluster', i
        idx = np.where(data.get_target() == i)
        X = data.get_data()[idx]
        plt.plot(X[:,-1], X[:,20], '.', color=c, label=labels[i])
    plt.legend()
    plt.xlim([0.4, 1.6])
    plt.ylim([10.5, 13])
    plt.title('Traffic flow clusters for ' + s)
    plt.savefig('pred_'+s+'_cluster.png')

if __name__ == '__main__':
    df = pd.read_csv('train_data.csv', index_col=0)
    df = df.drop('timestamp',1)
    X = df.as_matrix()

    train = pickle.load(open('data/train_data.p', 'rb'))
    test = pickle.load(open('data/test_data.p', 'rb'))

    # plot the true clusters
    #plot_clusters(train, s='train')
    #plot_clusters(test, s='test')

    train_model(train, test)


