#!/usr/bin/env python3
import pandas as pd
import os
import sys
import numpy as np
import yaml
from datetime import datetime, date

#def create_train_dateset(file):
def create_train_dateset(file: str) -> str:
    df_train = pd.read_csv(f"{file}train.csv")
    df_train["date"] = pd.to_datetime(df_train.date)

    df_store = pd.read_csv(f"{file}stores.csv")

    df_tran = pd.read_csv(f"{file}transactions.csv")
    df_tran["date"] = pd.to_datetime(df_tran.date)

    df_oil = pd.read_csv(f"{file}oil.csv")
    df_oil["date"] = pd.to_datetime(df_oil.date)

    df_holiday = pd.read_csv(f"{file}holidays_events.csv")
    df_holiday["date"] = pd.to_datetime(df_holiday.date)

    # missing data points in the daily oils data, processed by Linear Interpolation method
    # Reset index of oils dataframe
    df_oil = df_oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()

    # Linear Interpolation of Missing Values
    df_oil["dcoilwtico"] = np.where(df_oil["dcoilwtico"] == 0, np.nan, df_oil["dcoilwtico"])
    df_oil["interpolated_dcoilwtico"] = df_oil.dcoilwtico.interpolate()

    # feature engineering
    # merge files first
    # for the oil price which have an effect on store sales
    train0 = pd.merge(df_train, df_oil, how="left")
    # holiday will have effect on sales
    train1 = train0.merge(df_holiday, on='date', how='left')
    # Stores data will be merged with final prepareForTrain and test datasets
    train2 = train1.merge(df_store, on='store_nbr', how='left')
    train3 = train2.merge(df_tran, on=['date', 'store_nbr'], how='left')

    # There are several holiday events per year, so encoding 1 if occurs holiday else 0
    train3['holiday_flag'] = [1 if not val else 0 for val in train3['type_x'].isnull()]
    train3 = train3.drop(['type_x', 'locale_name', 'transferred'], axis=1)
    train3 = train3.rename(columns={'type_y': 'stores_type'})
    train3['transactions'] = train3['transactions'].fillna(0)
    train3['dcoilwtico'] = train3['dcoilwtico'].fillna(0)

    # change dtype and get the date col，adding them as new features
    train3['date'] = pd.to_datetime(train3['date']).dt.date
    train3['year'] = pd.to_datetime(train3['date']).dt.year
    train3['month'] = pd.to_datetime(train3['date']).dt.month
    train3['day'] = pd.to_datetime(train3['date']).dt.day

    final_train = train3.copy()
    # final_train.to_csv('/data/fortrain.csv', index=False, sep=';')

    # prepare data for training in LR,RF
    train_data = final_train.copy()
    datasimple = train_data.drop(['date', 'locale', 'description', 'dcoilwtico'], axis=1)
    datasimple = pd.get_dummies(datasimple, drop_first=True)
    datasimple['interpolated_dcoilwtico'] = datasimple['interpolated_dcoilwtico'].fillna(0)
    datasimple.to_csv('/data/simpletrainset.csv', index=False, sep=';')

    # train dataset for XGBoost model
    xgbtrain = final_train.copy()
    data_xgb = xgbtrain.drop(['date', 'locale', 'description', 'dcoilwtico'], axis=1)
    tep = final_train['date']
    data_xgb = pd.get_dummies(data_xgb, drop_first=True)
    data_xgb = pd.concat([tep, data_xgb], axis=1)
    data_xgb['interpolated_dcoilwtico'] = data_xgb['interpolated_dcoilwtico'].fillna(0)

    train_date = data_xgb['date'].unique()[-76:-15].tolist()
    valid_date = data_xgb['date'].unique()[-15:].tolist()
    data_xgb['is_train'] = data_xgb['date'].map(lambda x: x in train_date)
    data_xgb['is_valid'] = data_xgb['date'].map(lambda x: x in valid_date)
    data_xgb.to_csv('/data/trainset.csv', index=False, sep=';')


    return '/data/trainset.csv'

def create_test_dateset(file):
    df_test = pd.read_csv(f"{file}test.csv")
    df_test["date"] = pd.to_datetime(df_test.date)

    df_store = pd.read_csv(f"{file}stores.csv")

    df_tran = pd.read_csv(f"{file}transactions.csv")
    df_tran["date"] = pd.to_datetime(df_tran.date)

    df_oil = pd.read_csv(f"{file}oil.csv")
    df_oil["date"] = pd.to_datetime(df_oil.date)

    df_holiday = pd.read_csv(f"{file}holidays_events.csv")
    df_holiday["date"] = pd.to_datetime(df_holiday.date)

    # missing data points in the daily oils data, processed by Linear Interpolation method
    # Reset index of oils dataframe
    df_oil = df_oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()

    # Linear Interpolation of Missing Values
    df_oil["dcoilwtico"] = np.where(df_oil["dcoilwtico"] == 0, np.nan, df_oil["dcoilwtico"])
    df_oil["interpolated_dcoilwtico"] = df_oil.dcoilwtico.interpolate()

    # similar process for test dataset
    test_data = df_test.merge(df_oil, on='date', how='left')
    test_data = test_data.merge(df_holiday, on='date', how='left')
    test_data = test_data.merge(df_store, on='store_nbr', how='left')
    test_data = test_data.merge(df_tran, on=['date', 'store_nbr'], how='left')

    # There are several holiday events per year, so encoding 1 if occurs holiday else 0
    test_data['holiday_flag'] = [1 if not val else 0 for val in test_data['type_x'].isnull()]
    test_data = test_data.drop(['type_x', 'locale_name', 'transferred'], axis=1)
    test_data = test_data.rename(columns={'type_y': 'stores_type'})
    test_data['transactions'] = test_data['transactions'].fillna(0)
    test_data['dcoilwtico'] = test_data['dcoilwtico'].fillna(0)

    # change dtype and get the date col
    test_data['date'] = pd.to_datetime(test_data['date']).dt.date
    test_data['year'] = pd.to_datetime(test_data['date']).dt.year
    test_data['month'] = pd.to_datetime(test_data['date']).dt.month
    test_data['day'] = pd.to_datetime(test_data['date']).dt.day

    test_data_copy = test_data.copy()

    # test_data_copy.to_csv('/data/fortest.csv', index=False, sep=';')

    # prepare data for training in LR,RF
    simpletestset = test_data.copy()
    simpletestset = simpletestset.drop(['date', 'locale', 'description', 'dcoilwtico'], axis=1)
    simpletestset = pd.get_dummies(simpletestset, drop_first=True)
    simpletestset['interpolated_dcoilwtico'] = simpletestset['interpolated_dcoilwtico'].fillna(0)
    simpletestset.to_csv('/data/simpletestset.csv', index=False, sep=';')

    # test dataset for XGBoost
    xgbtest = test_data.copy()
    test_xgb = xgbtest.drop(['date', 'locale', 'description', 'dcoilwtico'], axis=1)
    tep_test = test_data['date']
    test_xgb = pd.get_dummies(test_xgb, drop_first=True)
    test_xgb = pd.concat([tep_test, test_xgb], axis=1)
    test_xgb['interpolated_dcoilwtico'] = test_xgb['interpolated_dcoilwtico'].fillna(0)
    test_xgb.to_csv('/data/testset.csv', index=False, sep=';')

    return '/data/testset.csv'

if __name__ == "__main__":

    command = sys.argv[1]

    functions = {
        "create_train_dateset": create_train_dateset,
        "create_test_dateset": create_test_dateset
    }

    argument = os.environ["FILE"]

    output = functions[command](argument)
    print("--> START CAPTURE")
    print(yaml.dump({"output": output}))
    print("--> END CAPTURE")