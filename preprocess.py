# Data Manipulation & Summarisation
import numpy as np 
from numpy import sqrt, abs, round
import pandas as pd

# debugging imports
from IPython import embed
import pdb

# Data Visualisation
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
color = sns.color_palette()
# %matplotlib inline

# Regex for working with text features
import re

# Datetime for working with datetime features
from datetime import timedelta, date

# Modules required for statistical tests
from scipy.stats import norm
from scipy.stats import t as t_dist
from scipy.stats import chi2_contingency

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ------------------------------------------------------------------------------------------------------------------------
# Simple script to quickly automate the preprocessing and less cluter the notebook, will aid with model building phase.
# ------------------------------------------------------------------------------------------------------------------------

# This function performs only labeling tranformation to work with sklearn. Bare bones preprocessing, expecting bad AUC...
def pre_baseline(df):
    print('=' * 100)
    print(f'[INFO] preprocessing script...Baseline preprocessing (minimal)')
    print("Only imputed NaNs and LabelEncoded strings categorical features")
    
    # impute missing bmi with mean 
    mean_val = df['bmi'].mean() 
    print(f'Imputing bmi with mean value: {mean_val}')
    df['bmi'] = df['bmi'].fillna(mean_val)
    # impute missing smoking_status with mode
    mode_val = df['smoking_status'].mode()[0] # forgetting that index was horrific, make sure to specify index 0
    print(f'Imputing smoking_status with mode value: {mode_val}')
    df['smoking_status'] = df['smoking_status'].fillna(mode_val)
    # label encode any features that are strings, sklearn cannot digest strings.
    df['Residence_type'] = LabelEncoder().fit_transform(df['Residence_type'])
    df['gender'] = LabelEncoder().fit_transform(df['gender'])
    df['ever_married'] = LabelEncoder().fit_transform(df['ever_married'])
    df['smoking_status'] = LabelEncoder().fit_transform(df['smoking_status'])
    df['work_type'] = LabelEncoder().fit_transform(df['work_type'])
    # print(df.head())    # no NaNs bc it's the train
    # print(df.tail())    # should see some NaN for stroke - bc append test under train.
    # segregate the train set from the test set and saved to cvs to be used in main notebook.
    dftrain=df[df['stroke'].isnull()!=True] 
    dftest=df[df['stroke'].isnull()==True]
    dftrain.to_csv('train_base.csv', index=False)
    dftest.to_csv('test_base.csv', index=False)
    return dftrain, dftest

# Add some new features that might be helpful
def pre_feature_eng_scale(df):
    print('=' * 100)
    print(f'[INFO] preprocessing script...Fully preprocessed')
    print("Imputed NaNs and LabelEncoded strings categorical features. Also added feature engineering")
    
    # Binning for categorical diabetic info - per resource above.
    bins = [0,100,180,400]
    labels = ['normal','pre-diabetic','diabetic']
    df['diabetic_ranges'] = pd.cut(df['avg_glucose_level'], bins= bins, labels=labels)
    #  Binning based on ranges for bmi
    bins = [0,18.5,25,30,100]
    labels = ['underweight','healthy','overweight','obese']
    df['obesity_indicator'] = pd.cut(df['bmi'], bins= bins, labels=labels)
    # from our EDA the max age was 82 and min was 0.08
    bins = [0, 5, 18 ,40 ,65, 98]
    labels = ['toddlers','kids','young_adults', 'getting_older', 'elderly']
    df['age_group'] = pd.cut(df['age'], bins= bins, labels=labels)
    # perhaps map elderly with heart_disease - to denote critical condition. This is a cool one.
    df['critical_condition'] = (df['age_group'] == 'elderly') &  (df['heart_disease'] == '1')
    
    # now we'll have to label encode the features crated above. 
    df['diabetic_ranges'] = LabelEncoder().fit_transform(df['diabetic_ranges'])
    df['obesity_indicator'] = LabelEncoder().fit_transform(df['obesity_indicator'])
    df['age_group'] = LabelEncoder().fit_transform(df['age_group'])
    df['critical_condition'] = LabelEncoder().fit_transform(df['critical_condition'])

    # log transform both features below, sklearn LogisticRegression assums Normal distributions for numerical data
    df['avg_glucose_level_log'] = np.log(df['avg_glucose_level'].values + 1)
    df['bmi_log'] = np.log(df['bmi'].values + 1)

    # scale to 0-1, easier for sklearn model to digest
    scaler = MinMaxScaler()
    scaler.fit(df[['avg_glucose_level', 'bmi', 'age']])
    df[['avg_glucose_level', 'bmi', 'age']] = scaler.transform(df[['avg_glucose_level', 'bmi', 'age']])

    # segregate the train set from the test set and saved to cvs to be used in main notebook.
    dftrain=df[df['stroke'].isnull()!=True] 
    dftest=df[df['stroke'].isnull()==True]
    dftrain.to_csv('train_full.csv', index=False)
    dftest.to_csv('test_full.csv', index=False)
    return dftrain, dftest

if __name__ == "__main__":
    # Reading Test Train & Data Dictionary
    train=pd.read_csv('train_AvX1lTZ.csv') 
    test=pd.read_csv('test_tERCnnc.csv')
    # let's combine both train and test into a single df so that any preprocessing is done two both train and test sets
    df=train.append(test,ignore_index=True)

    dftrain, dftest = pre_baseline(df.copy())
    dftrain, dftest = pre_feature_eng_scale(dftrain.append(dftest,ignore_index=True).copy())

