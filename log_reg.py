import pandas as pd
import numpy as np
import sys
import pickle
from util import timeit, create_cv_folds, get_cv_folds, plot_confusion_matrix
from folds import Fold
from scipy import interp
from sklearn import linear_model, metrics

'''
    http://sebastianraschka.com/faq/docs/evaluate-a-model.html
    http://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection
'''
# if sys.platform == 'darwin':
#     import matplotlib as mil
#     mil.use('TkAgg')
#
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     plot_on = True
#     print "Running OS X"
#
# elif sys.platform == 'linux' or sys.platform == 'linux2':
#     print "Running Linux. Plots are saved."
#     import matplotlib as mil
#     mil.use('Agg')
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     plot_on = False

@timeit
def train_model(cv_folds, train, test, num_folds=5):
    valid_err = []
    roc = []
    #p_C = [0.001, 0.004, 0.006]
    p_C = np.linspace(0.004, 0.009) #0.01
    p_multiclass = ['ovr', 'multinomial']
    p_solver = ['liblinear', 'newton-cg', 'lbfgs', 'sag']

    for c in p_C:                               # loop through the classifier's params
        for mc in p_multiclass:
            for s in p_solver:
                if mc == 'multinomial' and (s == 'liblinear' or s=='sag'):
                    break
                elif mc == 'ovr' and (s=='lbfgs' or s=='newton-cg'):
                    break  
                else:
                    print '----------fitting model with new params---------'
                    lr = linear_model.LogisticRegression(penalty='l2', C=c, solver =s, \
                                                        multi_class=mc, tol=0.001, n_jobs=-1)
                    #print lr
                    cv_err = []
                    for i in xrange(len(cv_folds)-1):
                        print '--------------fold-----------', i
                        folds = cv_folds[:i+1]
                        train_X, train_y = get_cv_folds(folds, num_folds)
                        
                        valid_X = cv_folds[i+1].get_data() 
                        valid_y = cv_folds[i+1].get_target()
                        lr = lr.fit(train_X, train_y)
                        preds = lr.predict(valid_X)
                        cv_err.append(metrics.f1_score(valid_y, preds, average='macro'))
                        print cv_err[i]
                    print 'fold avg validation error:', np.mean(cv_err)
                    valid_err.append( (lr, np.mean(cv_err)) )

    # select the best parameters
    best_classifier = max(valid_err, key= lambda x: x[1])
    print 'best classifier: \n', best_classifier

    # e = [1- x[1] for x in valid_err]
    # plt.plot(e, 'r--')
    # plt.ylabel('Error: 1- F1')
    # plt.xlabel('Model')
    # plt.title('Cross validation')

    # predict on test
    bclf = best_classifier[0].fit(train.get_data(), train.get_target())
    y_test_preds = bclf.predict(test.get_data())

    # print 'accuracy for test:'
    print metrics.classification_report(test.get_target(), y_test_preds)

    # plot the ROC on final classifier
    cm = metrics.confusion_matrix(test.get_target(), y_test_preds)
    plot_confusion_matrix(cm)


if __name__ == '__main__':
    # find the important features

    df = pd.read_csv('train_data.csv', index_col=0)
    df = df.drop('timestamp',1)
    X = df.as_matrix()

    cv_folds, train, test = create_cv_folds(X)
    #cv_folds = pickle.load(open('data/cv.p', 'rb'))
    #train = pickle.load(open('data/train_data.p', 'rb'))
    #test = pickle.load(open('data/test_data.p', 'rb'))
    train_model(cv_folds, train, test, num_folds=5)
    

