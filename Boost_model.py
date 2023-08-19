#%% Import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import requests
url = 'https://raw.githubusercontent.com/phuongnvp/BomonDuoclieu/Version-1.3/data.csv'
s=requests.get(url).content
df=pd.read_csv(io.StringIO(s.decode('utf-8')))
df.columns = ['Pub_CID' + str(i+1) if i<100 else col for i, col in enumerate(df.columns)]
df.columns = ['Pub_Excipient' + str(i+1) if 99<i<200 else col for i, col in enumerate(df.columns)]
df.shape
df["Outcome1"].value_counts()

#%% Define X and y
y = df['Outcome1'].values
X = df.drop(columns=["Outcome1", "API_CID", "Excipient_CID"], axis =1)
print(X.shape)
print(y.shape)

#%% Training set, validation set and test set
from sklearn.model_selection import train_test_split
X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size=.4, random_state=816)
X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=.5, random_state=817)

#%% Handle imbalanced data
from imblearn.over_sampling import SVMSMOTE
svmsmote = SVMSMOTE(random_state=42)
X_train_resampled, y_train_resampled = svmsmote.fit_resample(X_train, y_train)
X_train_resampled

#%% Random forest and XGBoost
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
model1 = AdaBoostClassifier(learning_rate = 0.7, n_estimators = 600)
model2 = RandomForestClassifier(n_estimators=100)
model3 = XGBClassifier(max_depth=5, learning_rate=0.5, n_estimators=100)

#%% Stack model
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
#ada = AdaBoostClassifier(n_estimators = 50, random_state=42)
lr = LogisticRegression(C=10)
stack_model = StackingClassifier(estimators = [('ada', model1), ('rf', model2), ('xgb', model3)], final_estimator = lr)

#%% Stack model - Fit data
class_weights = {0: 0.9, 1: 0.1}
stack_model.fit(X_train_resampled, y_train_resampled, sample_weight=[class_weights[i] for i in y_train_resampled])

#%% Stack model - Evaluation
from sklearn.metrics import accuracy_score, f1_score, precision_score
y_pred = stack_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
print("Accuracy:", accuracy)
print("F1-score:", f1)
print("Precision:", precision)

#%% Stack model - Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_val, y_pred, labels=stack_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=stack_model.classes_)
disp.plot()
plt.show()

#%% Stack model - Select threshold based on F1 score
from sklearn import metrics
yhat = stack_model.predict_proba(X_val)
probs = yhat[:,1]
thresholds = np.arange(0, 1, 0.001)
def to_labels(pos_probs, threshold):
 return (pos_probs >= threshold).astype('int')
scores = [f1_score(y_val, to_labels(probs, t)) for t in thresholds]
ix = np.argmax(scores)
threshold1 = thresholds[ix]
print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

#%% Stack model - Evaluate new threshold
y_pred_2 = (stack_model.predict_proba(X_val)[:,1] >= thresholds[ix]).astype(bool)
accuracy_2 = accuracy_score(y_val, y_pred_2)
f1_2 = f1_score(y_val, y_pred_2)
precision_2 = precision_score(y_val, y_pred_2)
print("Accuracy:", accuracy_2)
print("F1-score:", f1_2)
print("Precision:", precision_2)

#%% Stack model - Confusion matrix of new model
cm_2 = confusion_matrix(y_val, y_pred_2, labels=stack_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_2, display_labels=stack_model.classes_)
disp.plot()
plt.show()

#%% Stack model - Select threshold using GHOST algorithm (doi.org/10.1021/acs.jcim.1c00160)
import ghostml
from sklearn import metrics
import numpy as np

def calc_metrics(labels_test, test_probs, threshold):
    scores = [1 if x>=threshold else 0 for x in test_probs]
    auc = metrics.roc_auc_score(labels_test, test_probs)
    kappa = metrics.cohen_kappa_score(labels_test,scores)
    confusion = metrics.confusion_matrix(labels_test,scores, labels=list(set(labels_test)))
    print('thresh: %.2f, kappa: %.3f, AUC test-set: %.3f'%(threshold, kappa, auc))
    print(confusion)
    print(metrics.classification_report(labels_test,scores))
    return 

#%% Calculate threshold
train_probs = stack_model.predict_proba(X_train_resampled)[:,1]
thresholds = np.round(np.arange(0,1,0.001),2)
threshold2 = ghostml.optimize_threshold_from_predictions(y_train_resampled, train_probs, thresholds, ThOpt_metrics = 'Kappa') 
y_pred_3 = (stack_model.predict_proba(X_val)[:,1] >= threshold2).astype(bool)
calc_metrics(y_val, y_pred_3, threshold = threshold2)

#%% Stack model - Evaluate new threshold
accuracy_3 = accuracy_score(y_val, y_pred_3)
f1_3 = f1_score(y_val, y_pred_3)
precision_3 = precision_score(y_val, y_pred_3)
print("Accuracy:", accuracy_3)
print("F1-score:", f1_3)
print("Precision:", precision_3)

#%% Stack model - Confusion matrix of new model
cm_3 = confusion_matrix(y_val, y_pred_3, labels=stack_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_3, display_labels=stack_model.classes_)
disp.plot()
plt.show()

#%% Performance on test set
y_pred_4 = (stack_model.predict_proba(X_test)[:,1] >= threshold1).astype(bool)
calc_metrics(y_test, y_pred_4, threshold = threshold1)
accuracy_4 = accuracy_score(y_test, y_pred_4)
f1_4 = f1_score(y_test, y_pred_4)
precision_4 = precision_score(y_test, y_pred_4)
print("Accuracy:", accuracy_4)
print("F1-score:", f1_4)
print("Precision:", precision_4)
cm_4 = confusion_matrix(y_test, y_pred_4, labels=stack_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_4, display_labels=stack_model.classes_)
disp.plot()
plt.show()