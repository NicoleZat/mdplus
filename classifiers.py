from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from config import config
from imblearn.pipeline import make_pipeline, Pipeline

from sklearn.model_selection import GridSearchCV, StratifiedKFold

def randomForest(X_train, y_train, parameters, cv_scoring):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    return rf

def randomForestGS(X_train, y_train, parameters, cv_scoring):
    rf = RandomForestClassifier()
    gs = gridSearch(rf, parameters, cv_scoring, X_train, y_train)
    return gs

def logistic(X_train, y_train, parameters, cv_scoring):
    logit = LogisticRegression(solver='lbfgs', penalty='l2', class_weight='balanced')
    gs = gridSearch(logit, parameters, cv_scoring, X_train, y_train)
    return gs

def gridSearch(estimator, parameters, cv_scoring, X_train, y_train):
    skfgs = StratifiedKFold(n_splits=config.SPLIT)
    gs = GridSearchCV(estimator=estimator, param_grid=parameters, cv=skfgs, scoring=cv_scoring, verbose=config.VERBOSE, return_train_score=True)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, gs.cv_results_