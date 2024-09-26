import numpy as np
import pandas as pd
import os
from os import path
import pickle


def model_implentation(df):
    # Load feature names from a text file
    with open('trained_model/features.txt', 'r') as f:
        loaded_features = [line.strip() for line in f.readlines()]
    loaded_features[:25]
    df = df[loaded_features[:25]]

    # # Converting every Independent columns into numeric format 
    # df = df.apply(pd.to_numeric, errors='coerce') 

    # # Predicting Score of Logistic Refression
    # log_reg = pickle.load(open('trained_model/LogisticRegression.pkl', 'rb'))
    # log_pred = log_reg.predict(df)
    # log_pred_proba = log_reg.predict_proba(df)

    # # Predicting Score of Logistic Refression
    # svm = pickle.load(open('trained_model/SVMClassifier.pkl', 'rb'))
    # svm_pred = svm.predict(df)
    # svm_pred_proba = svm.predict_proba(df)

    # Predicting Score of Logistic Refression
    df_new = df[df.columns[:25]]
    dt = pickle.load(open('trained_model/DT.pkl', 'rb'))
    dt_pred = dt.predict(df_new)
    dt_pred_proba = dt.predict_proba(df_new)

    return dt_pred, dt_pred_proba


