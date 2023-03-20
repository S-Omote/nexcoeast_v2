import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
import pickle
import jpholiday
import os

cat_cols = ['road_code', 'start_code', 'end_code', 'section', 'direction', 'dayofweek', 'is_holiday']
num_cols = ['year', 'month', 'day', 'hour', 'search_1h', 'search_unspec_1d', 'KP', 'start_KP', 'end_KP', 'limit_speed',
                'OCC', 'allCars', 'speed', 'start_pref_code', 'end_pref_code', 'start_lat', 'end_lat', 'start_lng', 'end_lng',
                'start_degree', 'end_degree']
feature_cols = cat_cols + num_cols

def expand_datetime(df):
    if 'datetime' in df.columns:
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
    return df

def func_speed(x):
    if x.allCars == 0:
        return float(80)
    else:
        return x.speed

def extract_dataset(train_df, feature_cols):
    train_df0 = expand_datetime(train_df)
    train_df['section'] = train_df['start_code'].astype(str) + '_' + train_df['end_code'].astype(str)
    
    train_df['dayofweek'] = train_df['datetime'].dt.weekday
    train_df['date'] = train_df['datetime'].dt.date
    # dateから、祝日か否か判定する。祝日なら1。
    train_df['is_holiday'] = train_df['date'].map(jpholiday.is_holiday).astype(int)
    # allCars=0の場合、speedを80にする
    train_df['speed'] = train_df.apply(lambda x:func_speed(x), axis=1)
    
    
    with open('features/le_dict.pkl', 'rb') as web:
        le = pickle.load(web)
    for c in cat_cols:
        train_df[c] = le[c].transform(train_df[c])

    return train_df



class ScoringService(object):
    @classmethod
    def get_model(cls, model_path, inference_df):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            inference_df: Past data not subject to prediction.

        Returns:
            bool: The return value. True for success.
        """
        cls.n_splits = 5
        cls.model = {}
        for n in range(cls.n_splits):
            cls.model[n] = pickle.load(open(f'../model/lgb_fold{n}.pickle', 'rb'))

        data_path = os.path.join(model_path,"data")
        cls.speed_df = pd.read_csv(os.path.join(data_path,'speed_max_min_median_std.csv'))

        cls.data = inference_df

        return True


    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input: meta data of the sample you want to make inference from (DataFrame)

        Returns:
            prediction: Inference for the given input. Return columns must be ['datetime', 'start_code', 'end_code'](DataFrame).

        Tips:
            You can use past data by writing "cls.data".
        """
        prediction = extract_dataset(input, feature_cols)
        print('-------------------speed_df---------------', cls.speed_df.info())
        
        # max, minなどの情報をマージ
        #prediction = pd.merge(prediction, cls.speed_df, on=['start_code', 'end_code'])
        
        # 特徴量を絞る
        #prediction = prediction[feature_cols]
        #print(prediction.info())
        prediction['prediction'] = 0
        for n in range(cls.n_splits):
            prediction['prediction'] += cls.model[n].predict(prediction[feature_cols]) / cls.n_splits
        prediction.to_csv('prediction.csv')
        print('-------------------prediction---------------',prediction.info())
        prediction['prediction'] = prediction['prediction'].round()
        prediction['prediction'] = prediction['prediction'].astype(int)
        prediction['start_code'] = input["start_code"]
        prediction['end_code'] = input["end_code"]
        prediction['datetime'] = input["datetime"]
        prediction = prediction[['datetime', 'start_code', 'end_code', 'prediction']]
        #print(prediction['prediction'])

        return prediction