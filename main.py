from config import config
import classifiers
import results
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #key_nis = NIS record number
    #using these columns
    selectedColumns = ['age', 'amonth', 'aweekend', 'elective', 'female', 'hcup_ed', 'hosp_division',
                       'hosp_nis', 'i10_ndx', 'i10_npr', 'los', 'mdc', 'mdc_nopoa', 'nis_stratum', 'pay1', 'pl_nchs',
                       'race', 'totchg', 'tran_in', 'zipinc_qrtl', 'died']

    dataset2016 = pd.read_csv(config.DATAFILE2016)
    dataset2016 = dataset2016[selectedColumns]
    dataset2016 = dataset2016.dropna()

    dataset2017 = pd.read_csv(config.DATAFILE2017)
    dataset2017 = dataset2017[selectedColumns]
    dataset2017 = dataset2017.dropna()

    dataset2018 = pd.read_csv(config.DATAFILE2018)
    dataset2018 = dataset2018[selectedColumns]
    dataset2018 = dataset2018.dropna()

    dataset2019 = pd.read_csv(config.DATAFILE2019)
    dataset2019 = dataset2019[selectedColumns]
    dataset2019 = dataset2019.dropna()

    bigdata = pd.concat([dataset2016, dataset2017, dataset2018, dataset2019], ignore_index=True, sort=False)
    bigdata.age = bigdata.age.astype(int)
    bigdata.totchg = bigdata.totchg.astype(int)
    bigdata.tran_in = bigdata.tran_in.astype(int)
    bigdata.zipinc_qrtl = bigdata.zipinc_qrtl.astype(int)

    #set up
    length = len(bigdata.columns) - 1
    X = bigdata.iloc[:, 0:length]
    y = bigdata.iloc[:, length]

    # split data into sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TESTSIZE, random_state=config.SEED)

    #X.to_csv('test.csv')

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Logistic Regression
    #logisticParameters = [{'C': [0.001, 0.01, 0.1]}, {'max_iter': [999999]},
    #                      {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}]
    #classifier, bestParams, GSresults = classifiers.logistic(X_train, y_train, logisticParameters, config.CV_SCORING)
    # results.getResults(classifier, X_train, y_train, './results/logistic', bestParams, 'TRAIN', 0, GSresults)
    #results.getResults(classifier, X_test, y_test, './results/logistic', bestParams, 'TEST', 0, GSresults)

    #Random forest
    randomForestParameters = [{'estimators': [1000], 'criterion': ['entropy']}]
    #classifier, bestParams, GSresults = classifiers.randomForest2(X_train, y_train, randomForestParameters, config.CV_SCORING)
    classifier = classifiers.randomForest(X_train, y_train, randomForestParameters,
                                                                 config.CV_SCORING)
    #results.getResults(classifier, X_train, y_train, './results/randomForest2', bestParams, 'TRAIN', 0, GSresults)
    results.getResults(classifier, X_test, y_test, 'results/randomForest2')
    #results.getResults(classifier, X_test, y_test, './results/randomForest2', bestParams, 'TEST', 0, GSresults)