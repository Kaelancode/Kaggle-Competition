import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, \
RobustScaler, StandardScaler
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, \
PolynomialFeatures

from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_validate, GridSearchCV

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn import set_config
from sklearn.utils import estimator_html_repr

set_config(display='diagram')

path = os.getcwd()
data = pd.read_csv(path + '\\noel\data_ml.csv', index_col=0)

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

data[data.notnull().all(axis=1)].isnull().sum().all()
assert data[data.notnull().all(axis=1)].isnull().sum().all() == 0 , 'not null'

def run_pipeline():
    time_list = ['queuelen']
    scale_list = ['params', 'models']
    ohe_list = ['type', 'dayofweek']

    test = Pipeline([
                        ('SimpleImputer', SimpleImputer(missing_values=np.nan,
                                           strategy='constant',
                                           fill_value=1,
                                           add_indicator=False)),
                        #('KImputer',  KNNImputer(missing_values=np.nan,
                        #     add_indicator=False)),
                    ('scaler', PolynomialFeatures(interaction_only=True)),
            ])


    test2 = Pipeline([
                        ('KNNmputer',  KNNImputer(missing_values=np.nan,
                                 add_indicator=False)),
                        ('scaler', PolynomialFeatures(interaction_only=True)),
                        ('Sscale', StandardScaler()),
            ])


    p = ColumnTransformer([
                    ('test', test, ['type', 'params']),
                    ('test2', test2, ['models', 'trials']),
                ], remainder='passthrough')

    rf = RandomForestClassifier(n_estimators=100)
    #Test out p
    a = p.fit_transform(data)
    np.isnan(a).sum()

    pipe = Pipeline([('ohe', p), ('model', rf)])
    data_new = pipe.fit(x, y)
    print(data_new.predict(x))
    pipe
    plt.show()


    logoff = input('logoff: y/n')
    return pipe


if __name__ == '__main__':
    to_run = input('Run pipeline: y/n:')
    if to_run.lower() == 'y':
        run_pipeline()
    else:
        exit()
