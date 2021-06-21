import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

#violin plot of death vs creatinine_phosphokinase
plt.figure(1)
death_event = df['DEATH_EVENT'].values
cp = df['creatinine_phosphokinase'].values
sns.violinplot(x=death_event, y=cp)
plt.xlabel('DEATH_EVENT')
plt.ylabel('Creatinine phosphokinase level')
plt.show()
#violin plot of death vs ejection_fraction
plt.figure(2)
death_event = df['DEATH_EVENT'].values
cp = df['ejection_fraction'].values
sns.violinplot(x=death_event, y=cp)
plt.xlabel('DEATH_EVENT')
plt.ylabel('Ejection Fraction')
plt.show()
#violin plot of death vs serum_creatinine
plt.figure(3)
death_event = df['DEATH_EVENT'].values
cp = df['serum_creatinine'].values
sns.violinplot(x=death_event, y=cp)
plt.xlabel('DEATH_EVENT')
plt.ylabel('Serum creatinine levels')
plt.show()
#violin plot of death vs serum_sodium
plt.figure(4)
death_event = df['DEATH_EVENT'].values
cp = df['serum_sodium'].values
sns.violinplot(x=death_event, y=cp)
plt.xlabel('DEATH_EVENT')
plt.ylabel('Serum sodium levels')
plt.show()

#specifying features to test
test_features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 
                'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
                 'smoking']
test_data = df[test_features]

#specify target
target = df['DEATH_EVENT'].values

#split test and training groups
X_train, X_test, Y_train, Y_test = train_test_split(test_data, target, test_size = 0.25)

#K nearest neighbors prediction
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, Y_train)
print(f'K-nearest-neighbors: {knn.score(X_test, Y_test)}')

#Logistic Regression Prediction
lr = LogisticRegression()
lr.fit(X_train, Y_train)
print(f'Logistic Regression: {lr.score(X_test, Y_test)}')

#cross validation of knn
knn_cv_scores = cross_val_score(knn, X_train, Y_train, cv=3)
print('KNN Cross Validation Scores: ', knn_cv_scores)

#cross validation of logistic regression
lr_cv_scores = cross_val_score(lr, X_train, Y_train, cv=3)
print('Logistic Regression Cross Validation Scores: ', lr_cv_scores)
