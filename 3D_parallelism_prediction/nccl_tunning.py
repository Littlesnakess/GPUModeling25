# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV, train_test_split

import tools
import argparse

import os
import importlib


def xgboost_tuning(X, y):
    #FP32 
    # X = data[:, :-2]
    # y = data[:, y_index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 5, 10]
    }

    # Initialize XGBoost regressor
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # print("Best parameters found: ", grid_search.best_params_)
    
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    # Evaluate the average error percentage
    error = abs(predictions - y_test) / y_test
    return error.mean(), grid_search.best_params_    


def random_forest_tuning(X, y):
    #FP32 
    # X = data[:, :-2]
    # y = data[:, y_index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    }
    
    rf = RandomForestRegressor(random_state=42)
    # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    # print("Best parameters found: ", grid_search.best_params_)
    
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    # Evaluate the average error percentage
    error = abs(predictions - y_test) / y_test
    return error.mean(), grid_search.best_params_


def nccl_operator_tunning(data_path, config_path, column_indexes):

    data = pd.read_csv(data_path).to_numpy()
    X = data[:, column_indexes]
    y = data[:, -1]
    print(f'Working on {data_path}')

    dtree_error, dtree_parameter = random_forest_tuning(X, y)
    xgboost_error, xgboost_parameter = xgboost_tuning(X, y)

    if dtree_error < xgboost_error:
        best_model = 'rforest'
        best_parameters = dtree_parameter
        best_error = dtree_error
    else:
        best_model = 'xgboost'
        best_parameters = xgboost_parameter
        best_error = xgboost_error
 
    return best_model, best_parameters, best_error


def get_all_files_in_folder(folder_path):
    # Get all files in the folder
    file_names = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    return file_names


def operator_tunning_all(file_folder, columns_name):
    file_names = get_all_files_in_folder(file_folder)
    for file_name in file_names:
        name = file_name.split('.')[0]
        features = name.split('_')
        if len(features) >= 5:
            gpu_name = features[0]
            precision = features[-3]
            nodes = features[-2]
            gpus_per_node = features[-1]
            operator_name = '_'.join(features[1:-3])
            column_indexes = [0]
        else:
            gpu_name = features[0]
            precision = features[-1]
            nodes = 0
            gpus_per_node = 0
            operator_name = '_'.join(features[1:-1])
            column_indexes = [0, 1, 2]

        config_path = file_folder + '/' + gpu_name + '.csv'
        data_path = file_folder + '/' + file_name

        best_model, best_parameters, best_error = nccl_operator_tunning(data_path, config_path, column_indexes)

        result = [gpu_name, operator_name, precision, int(nodes), int(gpus_per_node)]
        result.append(best_model)
        result.append(best_parameters)
        result.append(best_error)

        tools.write_one_result_to_csv(config_path, columns_name, np.array(result))


# def nccl_tunning_test():
#     file_folder = 'Data/nccl_perlmutter/required'
#     # file_folder = 'Data/nccl_perlmutter/new'
#     columns_name = ['GPU', 'Function', 'Precision', 'Nodes', 'GPUsPerNode', 'Best_model', 'Best_config', 'Best_error%']
#     operator_tunning_all(file_folder, columns_name)


# def nccl_tunning_vista():

#     file_folder = './Data/nccl_vista'
#     # file_folder = 'Data/nccl_perlmutter/new'
#     columns_name = ['GPU', 'Function', 'Precision', 'Nodes', 'GPUsPerNode', 'Best_model', 'Best_config', 'Best_error%']
#     operator_tunning_all(file_folder, columns_name)


def nccl_tunning(file_folder):
    columns_name = ['GPU', 'Function', 'Precision', 'Nodes', 'GPUsPerNode', 'Best_model', 'Best_config', 'Best_error%']
    operator_tunning_all(file_folder, columns_name)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", default='', type=str,
                        help="Path of profilers' folder")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments() 
    nccl_tunning(args.target_path)


    # python nccl_tunning.py --target_path ./Data/nccl_perlmutter/required_renamed