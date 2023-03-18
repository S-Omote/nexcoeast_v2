import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
import pickle
import jpholiday

# 書き換え
cat_cols = ['road_code', 'start_code', 'end_code', 'section', 'direction', 'is_holiday', 'is_dayoff', 'is_dayoff_yesterday', 'is_dayoff_tomorrow']
num_cols = ['year', 'month', 'day', 'hour', 'search_1h', 'search_unspec_1d', 'KP', 'start_KP', 'end_KP', 'limit_speed',
                'OCC', 'allCars', 'speed', 'start_pref_code', 'end_pref_code', 'start_lat', 'end_lat', 'start_lng', 'end_lng',
                'start_degree', 'end_degree']
feature_cols = cat_cols + num_cols

# 書き換え
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
# 書き換え
def judge_dayoff(x):
    # 土日なら1を返す
    if  x.is_holiday == 1 or x.dayofweek > 4:
        return 1
    # 平日なら0を返す
    else:
        return 0
# 書き換え
def judge_yesterday(x):
    # 土日・祝日なら1を返す
    if x.is_holiday_yesterday == 1 or x.dayofweek_yesterday > 4:
        return 1
    # 平日なら0を返す
    else:
        return 0
def judge_tomorrow(x):
    # 土日・祝日なら1を返す
    if x.is_holiday_tomorrow == 1 or x.dayofweek_tomorrow > 4:
        return 1
    # 平日なら0を返す
    else:
        return 0

# 書き換え
def extract_dataset(train_df,feature_cols):
    train_df0 = expand_datetime(train_df)
    train_df['dayofweek'] = train_df['datetime'].dt.weekday
    train_df['date'] = train_df['datetime'].dt.date
    # dateから、祝日か否か判定する。祝日なら1。
    train_df['is_holiday'] = train_df['date'].map(jpholiday.is_holiday).astype(int)
    # 休日と平日の判定
    train_df['is_dayoff'] = train_df.apply(lambda x:judge_dayoff(x), axis=1)
    # 前日が休日か否か判定
    train_df['dayofweek_yesterday'] = (train_df['dayofweek']-1) % 7
    train_df['yesterday'] = train_df['date']
    train_df['yesterday'] -= pd.to_timedelta(1, 'd')
    train_df['is_holiday_yesterday'] = train_df['yesterday'].map(jpholiday.is_holiday).astype(int)
    train_df['is_dayoff_yesterday'] = train_df.apply(lambda x:judge_yesterday(x), axis=1)
    # 翌日が休日か否か判定
    train_df['dayofweek_tomorrow'] = (train_df['dayofweek']+1) % 7
    train_df['tomorrow'] = train_df['date']
    train_df['tomorrow'] += pd.to_timedelta(1, 'd')
    train_df['is_holiday_tomorrow'] = train_df['tomorrow'].map(jpholiday.is_holiday).astype(int)
    train_df['is_dayoff_tomorrow'] = train_df.apply(lambda x:judge_tomorrow(x), axis=1)
    
    train_df['section'] = train_df['start_code'].astype(str)+'_'+train_df['end_code'].astype(str)
    train_df = train_df[feature_cols]
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
        prediction = extract_dataset(input,feature_cols)
        prediction['prediction'] = 0
        for n in range(cls.n_splits):
            prediction['prediction'] += cls.model[n].predict(prediction[feature_cols]) / cls.n_splits
        prediction['prediction'] = prediction['prediction'].round()
        prediction['prediction'] = prediction['prediction'].astype(int)
        prediction['start_code'] = input["start_code"]
        prediction['end_code'] = input["end_code"]
        prediction['datetime'] = input["datetime"]
        prediction = prediction[['datetime', 'start_code', 'end_code', 'prediction']]

        return prediction