import numpy as np
import pandas as pd
import os
from os import path
import pickle
import features_preprocessing_pipeline as fpp
import model_implementation_pipeline as mip


def inferencing(questionid1, questionid2, question1, question2):

    data = {
        "qid1": [questionid1],
        "qid2": [questionid2],
        "question1": [question1],
        "question2": [question2]
    }
    df = pd.DataFrame(data).reset_index()
    df.columns = ["id", "qid1", "qid2", "question1", "question2"]

    # 1. Basic Data extraction
    df1 = df.copy()
    df1 = fpp.get_basic_feature(df1)

    # 2. NLP and Fuzzy features extraction
    df2 = df.copy()
    df2 = fpp.feature_extraction(df2)

    # # 3. TF_IDF weighted W2V feature Extraction
    # q1_w2v, q2_w2v = fpp.tf_idf_w2v(df)

    # df['q1_feats_m'] = list(q1_w2v)
    # df['q2_feats_m'] = list(q2_w2v)


    # df3_q1 = pd.DataFrame(df.q1_feats_m.values.tolist(), index= df.index)
    # df3_q2 = pd.DataFrame(df.q2_feats_m.values.tolist(), index= df.index)
    # df3_q1['id']=df3_q1.index
    # df3_q2['id']=df3_q1.index


    # Concating all the data frames
    # df1  = df1.merge(df2, on='id',how='left')
    # df2  = df3_q1.merge(df3_q2, on='id',how='left')
    result  = df1.merge(df2, on='id',how='left')

    # Model Implementation

    dt_pred, dt_pred_proba = mip.model_implentation(result)
    
    return dt_pred, dt_pred_proba
