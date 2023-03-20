'''
speedに関して
max,min,median,stdをcsvファイルとして保存する
'''
import pandas as pd

def main():
    df_train = pd.read_csv("C:/Users/brain/Desktop/nexcoeast_v2/train/train.csv")
    # speedに対して欠損値を埋める
    df_train['speed'] = pd.to_numeric(df_train["speed"], errors='coerce')
    
    result = pd.DataFrame()
    #df_train['section'] = df_train['start_code'].astype(str)+'_' + df_train['end_code'].astype(str)
    #print(df_train.info())
    df_max = pd.DataFrame(df_train.groupby(['start_code', 'end_code'])["speed"].max())
    df_max = df_max.rename(columns={"speed":"max_speed"})
    
    
    df_min = pd.DataFrame(df_train.groupby(['start_code', 'end_code'])["speed"].min())
    df_min = df_min.rename(columns={"speed":"min_speed"})
    
    df_median = pd.DataFrame(df_train.groupby(['start_code', 'end_code'])["speed"].median())
    df_median = df_median.rename(columns={"speed":"median_speed"})
    
    df_std = pd.DataFrame(df_train.groupby(['start_code', 'end_code'])["speed"].std())
    df_std = df_std.rename(columns={"speed":"std_speed"})
    #std_df = pd.read_csv('C:/Users/brain/Desktop/nexcoeast_v2/train/sample_std.csv')
    #print(std_df.info())
    

    #df_max.to_csv('sample_max.csv')
    #df_min.to_csv('sample_min.csv')
    #df_median.to_csv('sample_median.csv')
    
    df_max = df_max.round(2)
    #df_max['st'] = df_max["section"]
    df_max["min_speed"] = df_min['min_speed'].round(2)
    df_max["median_speed"] = df_median['median_speed'].round(2)
    df_max["std_speed"] = df_std['std_speed'].round(2)

    print(df_max.info())
    
    
    
    #csvファイルとして保存する
    df_max.to_csv("C:/Users/brain/Desktop/nexcoeast_v2/train/speed_max_min_median_std.csv")
    
    

if __name__ == '__main__':
    main()