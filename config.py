class Config(object):
    DATAFILE2016 = './NIS_2016_Core.csv'
    DATAFILE2017 = './NIS_2017_Core.csv'
    DATAFILE2018 = './NIS_2018_Core.csv'
    DATAFILE2019 = './NIS_2019_Core.csv'

    SEED = 1234
    SPLIT = 1
    TESTSIZE = .10
    NUMBER_OF_FOLDS = 10

    CV_SCORING = 'roc_auc'
    #CV_SCORING = 'accuracy'

    VERBOSE = 0
    SVM_KERNEL = 'rbf'
    #OVERSAMPLEMETHOD = 'SMOTE'


config = Config()