##### Import packages
# Basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modelling packages
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Other packages
from datetime import datetime
from collections import Counter

# To avoid warnings
import warnings
warnings.filterwarnings("ignore")


# # The objective of creating a prediction model based on Machine Learning Algorithms is to __check the probability of dying if you are COVID-19 confirmed__




##### Import data
# Check the csv's path before running it

df_est = pd.read_csv("CoordEstados.csv", encoding = "ISO-8859-1") # Mexican states data
df_cov = pd.read_csv("14.11.20 - COVID19MEXICO.csv", encoding = "ISO-8859-1") # Covid-19 data


# # Data Preparation




##### Create a new column with the time difference between been positive in COVID-19 and die
# If the person didn't die, time difference is 0

df_cov['FECHA_SINTOMAS'] = df_cov['FECHA_SINTOMAS'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df_cov['FECHA_DEF'] = df_cov['FECHA_DEF'].replace('9999-99-99', '2001-01-01')

df_cov['FECHA_DEF'] = df_cov['FECHA_DEF'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df_cov['DIFERENCIA'] = df_cov['FECHA_DEF'].sub(df_cov['FECHA_SINTOMAS'], axis=0)

df_cov['DIFERENCIA'] = df_cov['DIFERENCIA'] / np.timedelta64(1, 'D')
df_cov.loc[df_cov['DIFERENCIA']<0,'DIFERENCIA'] = 0





##### Create a new column with boolean value: 0 if the person doesn't have any disease, ≠ 0 otherwise
# Change the classification method from 1 yes 2 no to 1 yes 0 no. In this case, unknown values will be considered as no.

ill_name = ['DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA']
df_cov[ill_name] = df_cov[ill_name].replace([2,98],0)

df_cov['n_ENFERMEDADES'] = df_cov[ill_name].sum(axis = 1)





##### Create a new column (target) with boolean value: 0 if the person doesn't die, 1 otherwise

df_cov.loc[df_cov['DIFERENCIA']==0,'MORTALIDAD'] = 0
df_cov.loc[df_cov['DIFERENCIA']!=0,'MORTALIDAD'] = 1





##### Replacing missing values with NaNs

df_cov.replace([97,98,99],np.nan,inplace=True)
df_cov.isnull().sum()/len(df_cov)*100





##### Deleting columns with NaNs > 0.8

df_cov.drop(columns=['INTUBADO', 'MIGRANTE', 'UCI'], inplace = True)





##### Using SimpleImputer for missing values

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df_cov_imputer = imputer.fit_transform(df_cov)

df_cov = pd.DataFrame(df_cov_imputer, columns = df_cov.columns)
df_cov.isnull().sum()/len(df_cov)*100





##### Selecting all the columns to use to modelling (also the target)

features = list(df_cov)
remove = ['FECHA_ACTUALIZACION', 'ID_REGISTRO', 'FECHA_INGRESO','FECHA_SINTOMAS','FECHA_DEF','HABLA_LENGUA_INDIG','CLASIFICACION_FINAL',
          'PAIS_NACIONALIDAD','RESULTADO_LAB','PAIS_ORIGEN','NACIONALIDAD','EMBARAZO','DIFERENCIA','MORTALIDAD']

for col in remove:
    features.remove(col)


# # Modelling




##### Creation of x and y

X = df_cov[features].values.astype('int')
y = df_cov['MORTALIDAD'].values.astype('int')





##### Creation of X and y split -- train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)


# ## Random Forest




##### Random Forest

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

yhat = clf.predict(X_test)
print('Accuracy score: ',accuracy_score(y_test,yhat))





##### Classification Report

print(classification_report(y_test, yhat, digits=3))





##### Confussion Matrix

plot_confusion_matrix(clf, X_test, y_test,
                             display_labels=[0,1],
                             cmap=plt.cm.Blues,
                             normalize="true")
plt.title('Confussion matrix on how the \n model can predict mortality')
plt.show()





##### Feature Importance

features_importance = clf.feature_importances_
features_array = np.array(features)
features_array_ordered = features_array[(features_importance).argsort()[::-1]]
features_array_ordered

plt.figure(figsize=(16,10))
sns.barplot(y = features_array, x = features_importance, orient='h', order=features_array_ordered[:50])

plt.show()


# ## Logistic Regression




##### Logistic Regression

clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, y_train)

yhat = clf.predict(X_test)
print('Accuracy score: ',accuracy_score(y_test,yhat))





##### Classification Report

print(classification_report(y_test, yhat, digits=3))





##### Confussion Matrix

plot_confusion_matrix(clf, X_test, y_test,
                             display_labels=[0,1],
                             cmap=plt.cm.Blues,
                             normalize="true")
plt.title('Confussion matrix on how the \n model can predict mortality')
plt.show()


# ## UnderSampling




##### Undersampling to create sintetic data to improve class balance.
# Increase minority class size until its size represent 80% of major class size

undersampling = RandomUnderSampler(sampling_strategy=0.8) 
X_balance, y_balance = undersampling.fit_resample(X, y)
Counter(y_balance)





##### Creation of X and y split -- train and test applying undersampling

X_train, X_test, y_train, y_test = train_test_split(X_balance, y_balance, test_size=0.4)


# ## Random Forest with UnderSampling




##### Random Forest with UnderSampling

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

yhat = clf.predict(X_test)
print('Accuracy score: ',accuracy_score(y_test,yhat))





##### Classification Report

print(classification_report(y_test, yhat, digits=3))





##### Confussion Matrix

plot_confusion_matrix(clf, X_test, y_test,
                             display_labels=[0,1],
                             cmap=plt.cm.Blues,
                             normalize="true")
plt.title('Confussion matrix on how the \n model can predict mortality')
plt.show()





##### Feature Importance

features_importance = clf.feature_importances_
features_array = np.array(features)
features_array_ordered = features_array[(features_importance).argsort()[::-1]]
features_array_ordered

plt.figure(figsize=(16,10))
sns.barplot(y = features_array, x = features_importance, orient='h', order=features_array_ordered[:50])

plt.show()


# ## Logistic Regression with UnderSampling




##### Logistic Regression with UnderSampling

clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, y_train)

yhat = clf.predict(X_test)
print('Accuracy score: ',accuracy_score(y_test,yhat))





##### Classification Report

print(classification_report(y_test, yhat, digits=3))





##### Confussion Matrix

plot_confusion_matrix(clf, X_test, y_test,
                             display_labels=[0,1],
                             cmap=plt.cm.Blues,
                             normalize="true")
plt.title('Confussion matrix on how the \n model can predict mortality')
plt.show()

