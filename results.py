from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
import json

#def getResults(classifier, X_test, y_test, fileName, bestParams, set, foldNumber, GSresults):
def getResults(classifier, X_test, y_test, fileName):
    f = open(fileName, 'a')

    y_pred = classifier.predict(X_test)
    #f.write('\n' + 'SET: ' + set + '\n')
    #f.write('FOLD #: ' + str(foldNumber) + '\n')
    #f.write('GS Results: ' + '\n')
    #f.write(str(GSresults))
    #f.write('\n' + 'Best parameters: ' + '\n')
    #params = json.dumps(bestParams)
    #f.write(str(params))
    f.write('\n')
    f.write('Accuracy: ' + str(accuracy_score(y_test, y_pred)) + '\n')
    f.write(classification_report(y_test, y_pred) + '\n')
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    f.write('Confusion matrix results: ' + '\n')
    f.write('TP: ' + str(tp) + ' FP: ' + str(fp) + '\n')
    f.write('TN: ' + str(tn) + ' FN: ' + str(fn) + '\n')
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    f.write('Sensitivity: ' + str(sensitivity) + ' Specificity: ' + str(specificity) + '\n')
    roc = roc_auc_score(y_test, y_pred)
    f.write('ROC AUC: ' + str(roc) + '\n')
    ppv = tp/(tp+fp)
    npv = tn/(tn+fn)
    f.write('PPV: ' + str(ppv) + ' NPV: ' + str(npv) + '\n')
    f1 = f1_score(y_test, y_pred)
    f.write('F1 score: ' + str(f1) + '\n')
    f.close()