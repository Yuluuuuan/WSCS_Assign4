import pandas as pd
import matplotlib.pyplot as plt



#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys


def grouped(df, key, freq, col):
    """ GROUP DATA WITH CERTAIN FREQUENCY """
    df_grouped = df.groupby([pd.Grouper(key=key, freq=freq)]).agg(mean=(col, 'mean'))
    df_grouped = df_grouped.reset_index()
    return df_grouped


def add_time(df, key, freq, col):
    """ ADD COLUMN 'TIME' TO DF """
    df_grouped = grouped(df, key, freq, col)
    df_grouped['time'] = np.arange(len(df_grouped.index))
    column_time = df_grouped.pop('time')
    df_grouped.insert(1, 'time', column_time)
    return df_grouped

def draw_line(csv_path,output_path):
    df_holidays = pd.read_csv('data/holidays_events.csv', header=0)
    df_oil = pd.read_csv(f'{csv_path}/oil.csv', header=0)
    df_stores = pd.read_csv(f'{csv_path}/stores.csv', header=0)
    df_trans = pd.read_csv(f'{csv_path}/transactions.csv', header=0)
    df_train = pd.read_csv(f'{csv_path}/train.csv', header=0)
    df_test = pd.read_csv(f'{csv_path}/test.csv', header=0)

    df_holidays['date'] = pd.to_datetime(df_holidays['date'], format="%Y-%m-%d")
    df_oil['date'] = pd.to_datetime(df_oil['date'], format="%Y-%m-%d")
    df_trans['date'] = pd.to_datetime(df_trans['date'], format="%Y-%m-%d")
    df_train['date'] = pd.to_datetime(df_train['date'], format="%Y-%m-%d")
    df_test['date'] = pd.to_datetime(df_test['date'], format="%Y-%m-%d")

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(25, 15))
    df_oil.plot.line(x="date", y="dcoilwtico", color='b', title="dcoilwtico", ax=axes, rot=0)
    plt.show()

    df_grouped_train_w = add_time(df_train, 'date', 'W', 'sales')
    df_grouped_train_m = add_time(df_train, 'date', 'M', 'sales')
    df_grouped_trans_w = add_time(df_trans, 'date', 'W', 'transactions')
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(30, 20))

    # TRANSACTIONS (WEEKLY)
    axes[0].plot('date', 'mean', data=df_grouped_trans_w, color='grey', marker='o')
    axes[0].set_title("Transactions (grouped by week)", fontsize=20)

    # SALES (WEEKLY)
    axes[1].plot('time', 'mean', data=df_grouped_train_w, color='0.75')
    axes[1].set_title("Sales (grouped by week)", fontsize=20)
    # linear regression
    axes[1] = sns.regplot(x='time',
                          y='mean',
                          data=df_grouped_train_w,
                          scatter_kws=dict(color='0.75'),
                          ax=axes[1])

    # SALES (MONTHLY)
    axes[2].plot('time', 'mean', data=df_grouped_train_m, color='0.75')
    axes[2].set_title("Sales (grouped by month)", fontsize=20)
    # linear regression
    axes[2] = sns.regplot(x='time',
                          y='mean',
                          data=df_grouped_train_m,
                          scatter_kws=dict(color='0.75'),
                          line_kws={"color": "red"},
                          ax=axes[2])

    plt.show()
    fig.savefig("line.png")
    plt.close()

    return output_path



def draw_pre(csv_path:str,output_path:str)->str:
    store_sales = pd.read_csv(f"{csv_path}/train.csv", header=0)
    pre_sales = pd.read_csv(f"{csv_path}/allinfo_XGBoost.csv", header=0)

    store_sales['date'] = pd.to_datetime(store_sales['date'], format="%Y-%m-%d")
    pre_sales['date'] = pd.to_datetime(pre_sales['date'], format="%Y-%m-%d")
    average_sales = store_sales.groupby('date').mean()['sales']
    pre_sales = pre_sales.groupby('date').mean()['sales']

    fig, axes = plt.subplots(figsize=(80, 40))

    ax = average_sales.plot(alpha=1, title="Average Sales", ylabel="items sold")
    ax = pre_sales.plot(ax=ax, label="Forecast", color='C3')
    ax.legend()
    plt.show()
    fig.savefig("prediction.png")
    plt.close()
    return output_path


if __name__ == "__main__":
    # store_sales = pd.read_csv('data/train.csv', header=0)
    # pre_sales=pd.read_csv("data/allinfo_XGBoost.csv", header=0)
    # # store_sales['date'] = pd.to_datetime(store_sales['date'], format="%Y-%m-%d")
    # # pre_sales['date'] = pd.to_datetime(pre_sales['date'], format="%Y-%m-%d")
    # # average_sales = store_sales.groupby('date').mean()['sales']
    # # pre_sales = pre_sales.groupby('date').mean()['sales']
    fig, axes = plt.subplots( figsize=(80, 40))
    #draw_line('data', "data")
    draw_pre('data', 'data')

