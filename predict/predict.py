#!/usr/bin/env python3
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import os
import sys
import numpy as np
import yaml


def run_LR(input_path: str, output_path:str) -> str:
    LR_train = pd.read_csv(f"{input_path}/simpletrainset.csv", sep=';')
    LR_test = pd.read_csv(f"{output_path}/simpletestset.csv", sep=';')
    # begin training process
    X = LR_train.drop('sales', axis=1)
    y = LR_train['sales']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1234)

    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    pre_X = LR_test.copy()
    pre_y = model.predict(pd.DataFrame(pre_X))

    preds_LR = pd.DataFrame({
        "id": pre_X["id"],
        "sales": pre_y
    })
    preds_LR.to_csv(f"{output_path}/submission_LR.csv", index=False)

    return "/data/submission_LR.csv"



def run_RF(input_path: str, output_path:str) -> str:
    RF_train = pd.read_csv(f"{input_path}/simpletrainset.csv", sep=';')
    RF_test = pd.read_csv(f"{input_path}/simpletestset.csv", sep=';')
    # begin training process
    X = RF_train.drop('sales', axis=1)
    y = RF_train['sales']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1234)

    # Random Forest
    RF = RandomForestRegressor(random_state=12, n_jobs=-1, max_depth=30, max_features='log2', max_leaf_nodes=20,
                               verbose=2)
    RF.fit(X_train, y_train)

    pre_X = RF_test.copy()
    pre_y_1 = RF.predict(pd.DataFrame(pre_X))

    preds_LR = pd.DataFrame({
        "id": pre_X["id"],
        "sales": pre_y_1
    })
    preds_LR.to_csv(f"{output_path}/submission_RF.csv", index=False)
    return "/data/submission_RF.csv"

#def run_xgboost(trainset, testset,subformat):
def run_xgboost(input_path:str,output_path:str)->str:
    xgb_params = {
        'tree_method': 'hist',
        'gpu_id': 0,
        'booster': 'gbtree',
        # 'predictor': 'gpu_predictor',
        'verbosity': 2,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 2021,
        'learning_rate': 0.009,
        'subsample': 0.99,
        'colsample_bytree': 0.80,
        'reg_alpha': 10.0,
        'reg_lambda': 0.18,
        'min_child_weight': 47,
    }

    train_set = pd.read_csv(f"{input_path}/trainset.csv", sep=';')
    X_train = train_set.drop(['date', 'sales', 'year'], axis=1)
    y = np.log(train_set['sales'] + 1)

    test_set = pd.read_csv(f"{input_path}/testset.csv", sep=';')
    X_test = test_set.drop(['date', 'year'], axis=1)
    #start = time.time()
    # extract train and valid dataset
    trn_idx = X_train[X_train['is_train'] == True].index.tolist()
    val_idx = X_train[X_train['is_valid'] == True].index.tolist()

    X_tr = X_train.loc[trn_idx, :].drop(['is_train', 'is_valid'], axis=1)
    X_val = X_train.loc[val_idx, :].drop(['is_train', 'is_valid'], axis=1)
    y_tr = y[trn_idx]
    y_val = y[val_idx]

    xgb_train = xgb.DMatrix(X_tr, y_tr)
    xgb_valid = xgb.DMatrix(X_val, y_val)
    evallist = [(xgb_train, 'train'), (xgb_valid, 'eval')]
    evals_result = dict()
    #dtrain = xgb.DMatrix(X_train, y)
    model = xgb.train(params=xgb_params, dtrain=xgb_train, evals=evallist, evals_result=evals_result,
                      verbose_eval=5000, num_boost_round=100000, early_stopping_rounds=100)

    #xgb_oof = np.zeros(y_val.shape[0])
    xgb_oof = model.predict(xgb_valid, iteration_range=(0, model.best_iteration))

    xgb_test = xgb.DMatrix(X_test)
    xgb_pred = pd.Series(model.predict(xgb_test, iteration_range=(0, model.best_iteration)),
                         name='xgb_pred')
    sub = pd.read_csv(f"{input_path}sample_submission.csv", sep=',')
    #sub.to_csv('.\\test.csv', index=False, sep=';')
    sub['sales'] = np.exp(xgb_pred) - 1
    sub_process = sub.copy()

    sub_process[sub_process < 0] = 0
    sub_process.to_csv(f'{output_path}/XGB_Model_final.csv', index=False, sep=';')
    #elapsed = time.time() - start
    #mse = mean_squared_error(y_val, xgb_oof, squared=False)
    #     rmsle = np.sqrt(mean_squared_log_error(y_true=y_val, y_pred=xgb_oof))
    #mae = mean_absolute_error(y_val, xgb_oof)
    #print(f" rmse: {mse:.6f}\n mae: {mae:.6f} , elapsed time: {elapsed:.2f}sec\n")
    return '/data/XGB_Model_final.csv'

#def final_result(XGBsub, formerge, testset):
def final_result(input_path:str, output_path:str)->str:
    df_test = pd.read_csv(f"{input_path}/XGB_Model_final.csv", sep=';')
    test_set = pd.read_csv(f"{input_path}/testset.csv", sep=';')
    #print(test_set.columns)
    X_test = test_set.drop(columns=['date', 'year'],inplace=True)
    test_merge = pd.concat([df_test, X_test], axis=1)
    test_merge = test_merge.loc[:, ~test_merge.columns.duplicated()]
    sub_process = pd.read_csv(f"{input_path}/testset.csv", sep=';')
    sub_info = pd.merge(sub_process,test_merge, left_on = 'id',right_index=True)
    sub_info.to_csv(f"{output_path}/allinfo_XGBoost_info.csv", index=False, sep=';')
    return "/data/allinfo_XGBoost_info.csv"


if __name__ == "__main__":
    command = sys.argv[1]
    input_path = os.environ["INPUT_PATH"]
    output_path = os.environ["OUTPUT_PATH"]
    functions = {
        "run_LR": run_LR,
        "run_RF": run_RF,
        "run_xgboost": run_xgboost,
        "final_result": final_result
    }
    output = functions[command](input_path,output_path)
    print(yaml.dump({"output": output}))







