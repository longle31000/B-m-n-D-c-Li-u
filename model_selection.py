#%%
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import io
import requests
url = 'https://raw.githubusercontent.com/phuongnvp/BomonDuoclieu/main/data.csv'
s=requests.get(url).content
df=pd.read_csv(io.StringIO(s.decode('utf-8')))
print(df)
#%%
y = df.iloc[:,-1].values
X = df.iloc[:,0:200].values
print(y)
print(X)
#%% Training set, validation set and test set
from sklearn.model_selection import train_test_split
X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size=.4, random_state=816)
X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=.5, random_state=817)
#%%
from imblearn.over_sampling import SVMSMOTE
svmsmote = SVMSMOTE(random_state=42)
X_train_resampled, y_train_resampled = svmsmote.fit_resample(X_train, y_train)
X_train_resampled
#%%
#Random forest
from sklearn.ensemble import RandomForestClassifier

param_grid_RF = {
        'n_estimators': [10, 50, 100],
}

grid_search_RF = GridSearchCV(RandomForestClassifier(), param_grid = param_grid_RF, 
                           scoring = 'accuracy',
                           cv = 5, n_jobs = -1, verbose = 0)
# Fit the grid search to the data
grid_search_RF.fit(X_train, y_train)
y_pred_RF = grid_search_RF.predict(X_val)
print('Best parameters: ',grid_search_RF.best_params_)
print('Accuracy: ',accuracy_score(y_val,y_pred_RF))
print(classification_report(y_val,y_pred_RF))
#%%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_val, y_pred_RF, labels=grid_search_RF.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search_RF.classes_)
disp.plot()
plt.show()
#%%
#Desicion tree
from sklearn.tree import DecisionTreeClassifier
param_grid_DT = {
        'max_depth': [5, 15, 25],
}

grid_search_DT = GridSearchCV(DecisionTreeClassifier(), param_grid = param_grid_DT, 
                           scoring = 'accuracy',
                           cv = 5, n_jobs = -1, verbose = 0)
# Fit the grid search to the data
grid_search_DT.fit(X_train, y_train)
y_pred_DT = grid_search_DT.predict(X_val)
print('Best parameters: ',grid_search_DT.best_params_)
print('Accuracy: ',accuracy_score(y_val,y_pred_DT))
print(classification_report(y_val,y_pred_DT))
#%%
cm = confusion_matrix(y_val, y_pred_DT, labels=grid_search_DT.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search_DT.classes_)
disp.plot()
plt.show()
#%%
#Support vector machine
from sklearn.svm import SVC
param_grid_SVC = {
    'gamma': [0.001, 0.01, 0.1],
    'C': [1, 10, 100],
}

# Instantiate the grid search model
grid_search_SVC = GridSearchCV(SVC(), param_grid = param_grid_SVC, 
                           scoring = 'accuracy',
                           cv = 5, n_jobs = -1, verbose = 0)
# Fit the grid search to the data
grid_search_SVC.fit(X_train, y_train)
print('Best parameters: ',grid_search_SVC.best_params_)
y_pred_SVC = grid_search_SVC.predict(X_val) 
# print classification report 
print('Accuracy: ',accuracy_score(y_val,y_pred_SVC))
print(classification_report(y_val,y_pred_SVC))
#%%
cm = confusion_matrix(y_val, y_pred_SVC, labels=grid_search_SVC.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search_SVC.classes_)
disp.plot()
plt.show()
#%%
#Multi-layer Perceptron
from sklearn.neural_network import MLPClassifier

param_grid_MLP = {
    'hidden_layer_sizes': [(100,), (300,), (500,)],
    'activation': ['relu'],
    'solver': ['adam'],
    'momentum': [0.1, 0.5, 0.9],
    'learning_rate': ['adaptive'],
}
# Instantiate the grid search model
grid_search_MLP = GridSearchCV(MLPClassifier(), param_grid = param_grid_MLP, 
                           scoring = 'accuracy',
                           cv = 5, n_jobs = -1, verbose = 0)
# Fit the grid search to the data
grid_search_MLP.fit(X_train, y_train)
print('Best parameters: ',grid_search_MLP.best_params_)
y_pred_MLP = grid_search_MLP.predict(X_val) 
# print classification report 
print('Accuracy: ',accuracy_score(y_val,y_pred_MLP))
print(classification_report(y_val,y_pred_MLP))
#%%
cm = confusion_matrix(y_val, y_pred_MLP, labels=grid_search_MLP.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search_MLP.classes_)
disp.plot()
plt.show()
#%%
#k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

param_grid_kNN = {
    'n_neighbors': np.arange(3, 16),
}
# Instantiate the grid search model
grid_search_kNN = GridSearchCV(KNeighborsClassifier(), param_grid = param_grid_kNN, 
                           scoring = 'accuracy',
                           cv = 5, n_jobs = -1, verbose = 0)
# Fit the grid search to the data
grid_search_kNN.fit(X_train, y_train)
print('Best parameters: ',grid_search_kNN.best_params_)
y_pred_kNN = grid_search_kNN.predict(X_val) 
# print classification report 
print('Accuracy: ',accuracy_score(y_val,y_pred_kNN))
print(classification_report(y_val,y_pred_kNN))
#%%
cm = confusion_matrix(y_val, y_pred_kNN, labels=grid_search_kNN.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search_kNN.classes_)
disp.plot()
plt.show()
#%%
#Logistic Regression
from sklearn.linear_model import LogisticRegression
param_grid_LR = {
    'C': [0.01, 1, 10, 100],
}
# Instantiate the grid search model
grid_search_LR = GridSearchCV(LogisticRegression(), param_grid = param_grid_LR, 
                           scoring = 'accuracy',
                           cv = 5, n_jobs = -1, verbose = 0)
# Fit the grid search to the data
grid_search_LR.fit(X_train, y_train)
print('Best parameters: ',grid_search_LR.best_params_)
y_pred_LR = grid_search_LR.predict(X_val) 
# print classification report 
print('Accuracy: ',accuracy_score(y_val,y_pred_LR))
print(classification_report(y_val,y_pred_LR))
#%%
cm = confusion_matrix(y_val, y_pred_LR, labels=grid_search_LR.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search_LR.classes_)
disp.plot()
plt.show()
#%%
#Naive Bayes
from sklearn.naive_bayes import GaussianNB, BernoulliNB
GNB = GaussianNB()
BNB = BernoulliNB()
GNB.fit(X_train, y_train)
y_pred_GNB = GNB.predict(X_val) 
# print classification report 
print('Accuracy: ',accuracy_score(y_val,y_pred_GNB))
print(classification_report(y_val,y_pred_GNB))
print('---------------')
BNB.fit(X_train, y_train)
y_pred_BNB = BNB.predict(X_val) 
# print classification report 
print('Accuracy: ',accuracy_score(y_val,y_pred_BNB))
print(classification_report(y_val,y_pred_BNB))
#%%
cm = confusion_matrix(y_val, y_pred_GNB, labels=GNB.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GNB.classes_)
disp.plot()
plt.show()
print('---------------')
cm = confusion_matrix(y_val, y_pred_BNB, labels=BNB.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=BNB.classes_)
disp.plot()
plt.show()

#%%
#XGBoost
from xgboost import XGBClassifier
param_grid_XGB = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.3, 0.5],
}
# Instantiate the grid search model
grid_search_XGB = GridSearchCV(XGBClassifier(), param_grid = param_grid_XGB, 
                           scoring = 'accuracy',
                           cv = 5, n_jobs = -1, verbose = 0)
# Fit the grid search to the data
grid_search_XGB.fit(X_train, y_train)
print('Best parameters: ',grid_search_XGB.best_params_)
y_pred_XGB = grid_search_XGB.predict(X_val) 
# print classification report 
print('Accuracy: ',accuracy_score(y_val,y_pred_XGB))
print(classification_report(y_val,y_pred_XGB))
#%%
cm = confusion_matrix(y_val, y_pred_XGB)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()