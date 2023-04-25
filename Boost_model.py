#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("F:\Phuong\Project\API Excipient interaction\Final\Streamlit app\BomonDuoclieu\data.csv")
df.columns = ['Pub_CID' + str(i+1) if i<100 else col for i, col in enumerate(df.columns)]
df.columns = ['Pub_Excipient' + str(i+1) if 99<i<200 else col for i, col in enumerate(df.columns)]
df.shape
df["Outcome1"].value_counts()

#%%
y = df['Outcome1'].values
X = df.drop(columns=["Outcome1", "API_CID", "Excipient_CID"], axis =1)
print(X.shape)
print(y.shape)

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=816)
#%%
from imblearn.over_sampling import SVMSMOTE
svmsmote = SVMSMOTE(random_state=42)
X_train_resampled, y_train_resampled = svmsmote.fit_resample(X_train, y_train)
X_train_resampled

#%%
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
model2 = RandomForestClassifier(n_estimators=100)
model3 = XGBClassifier(max_depth=7, learning_rate=0.3, n_estimators=100)

#%%
from sklearn.ensemble import VotingClassifier
voting_model = VotingClassifier(estimators=[('rf', model2), ('xgb', model3)], voting='soft')
boost_model = AdaBoostClassifier(base_estimator= voting_model, n_estimators=15, algorithm='SAMME')

#%%
class_weights = {0: 0.9, 1: 0.1}
boost_model.fit(X_train_resampled, y_train_resampled, sample_weight=[class_weights[i] for i in y_train_resampled])

#%% Mô hình mặc định
from sklearn.metrics import accuracy_score, f1_score, precision_score
y_pred = boost_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print("Độ chính xác:", accuracy)
print("F1-score:", f1)
print("Precision:", precision)
#%%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred, labels=boost_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=boost_model.classes_)
disp.plot()
plt.show()

#%% Chọn threshold dựa vào F1 score
from sklearn import metrics
yhat = boost_model.predict_proba(X_test)
probs = yhat[:,1]
thresholds = np.arange(0, 1, 0.001)
def to_labels(pos_probs, threshold):
 return (pos_probs >= threshold).astype('int')
scores = [f1_score(y_test, to_labels(probs, t)) for t in thresholds]
ix = np.argmax(scores)
print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

#%%
y_pred_2 = (boost_model.predict_proba(X_test)[:,1] >= thresholds[ix]).astype(bool)
accuracy_2 = accuracy_score(y_test, y_pred_2)
f1_2 = f1_score(y_test, y_pred_2)
precision_2 = precision_score(y_test, y_pred_2)
print("Độ chính xác:", accuracy_2)
print("F1-score:", f1_2)
print("Precision:", precision_2)

#%%
cm_2 = confusion_matrix(y_test, y_pred_2, labels=boost_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_2, display_labels=boost_model.classes_)
disp.plot()
plt.show()

#%% Chọn threshold bằng thuật toán GHOST (doi.org/10.1021/acs.jcim.1c00160)
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

#%%
train_probs = boost_model.predict_proba(X_train_resampled)[:,1]
thresholds = np.round(np.arange(0,1,0.001),2)
threshold1 = ghostml.optimize_threshold_from_predictions(y_train_resampled, train_probs, thresholds, ThOpt_metrics = 'Kappa') 
y_pred_3 = (boost_model.predict_proba(X_test)[:,1] >= threshold1).astype(bool)
calc_metrics(y_test, y_pred_3, threshold = threshold1)

#%%
accuracy_3 = accuracy_score(y_test, y_pred_3)
f1_3 = f1_score(y_test, y_pred_3)
precision_3 = precision_score(y_test, y_pred_3)
print("Độ chính xác:", accuracy_3)
print("F1-score:", f1_3)
print("Precision:", precision_3)

#%%
cm_3 = confusion_matrix(y_test, y_pred_3, labels=boost_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_3, display_labels=boost_model.classes_)
disp.plot()
plt.show()

#%%
import joblib
joblib.dump(boost_model, 'F:\Phuong\Project\API Excipient interaction\Final\model100.pkl')
