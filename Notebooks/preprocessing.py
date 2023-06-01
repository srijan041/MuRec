import pandas as pd
import numpy as np
from sklearn import preprocessing as prp
from pandas.api.types import is_numeric_dtype

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns




dataset = pd.read_csv('/mnt/4CB2D623B2D610F6/Projects/Extra/MLProject/test-apps/StreamLit-app/music-recommendation-app/Datasets/dataset.csv')


dataset = dataset[['track_id', 'artists', 'track_genre', 'mode' , 'key' ,'popularity' ,'danceability', 'energy', 'speechiness', 'acousticness', 'liveness', 'valence', 'tempo' ]]





dataset.drop_duplicates(subset=['track_id'], inplace = True)


dataset.dropna(inplace = True)
dataset.reset_index(drop = True, inplace = True)


def generate_list(col_name):
  return list(dataset[col_name].apply(lambda x : [i.replace("-", "").replace(" ", "").replace(".","") for i in x]))

dataset['artists'] = dataset['artists'].apply(lambda x:x.lower().split(';'))
dataset['artists'] = generate_list('artists')


dataset['track_genre'] = dataset['track_genre'].apply(lambda x: x.split())
dataset['track_genre'] = generate_list('track_genre')


def ohe_prep(df, column, new_name): 
    
    tf_df = pd.get_dummies(df[column])
    feature_names = tf_df.columns
    tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
    tf_df.reset_index(drop = True, inplace = True)    
    return tf_df


def create_feature_set(df, float_cols):
    
    tfidf = TfidfVectorizer()
    tfidf_matrix =  tfidf.fit_transform(dataset['track_genre'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
    genre_df.reset_index(drop = True, inplace=True)

    key_ohe = ohe_prep(dataset, 'key','key') * 0.5
    mode_ohe = ohe_prep(dataset, 'mode','mode') * 0.5

    pop = dataset[['popularity']].reset_index(drop = True)
    scaler = MinMaxScaler()
    pop_scaled = pd.DataFrame(scaler.fit_transform(pop), columns = pop.columns) * 0.2 

    floats = dataset[float_cols].reset_index(drop = True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2

    final = pd.concat([genre_df,floats_scaled, pop_scaled, key_ohe, mode_ohe], axis = 1)

    final['track_id']=dataset['track_id'].values
    
    return final


float_cols = dataset.dtypes[dataset.dtypes == 'float64'].index.values


complete_feature_set = create_feature_set(dataset, float_cols=float_cols)
