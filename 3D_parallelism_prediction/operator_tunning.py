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


def get_map_function(module_name, function_name):
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def operator_tunning(file_folder, config_fold, gpu_name, operator_name, propagation, precision, columns_name):
    # config_fold = './Model_config'
    os.makedirs(config_fold, exist_ok=True)
    config_path =  config_fold + '/' + gpu_name + '.csv'
    if precision == '':
        file_name = gpu_name + '_' + operator_name + '.csv'
    else:
        file_name = gpu_name + '_' + operator_name + '_' + precision + '.csv'

    file_path = os.path.join(file_folder, file_name)
    data = pd.read_csv(file_path).to_numpy()
    if propagation == 'dur':
        X = data[:, 0:-1]
    else:    
        X = data[:, 0:-2]

    map_function = get_map_function('layer_input_to_predictor_input', operator_name)
    X = np.apply_along_axis(map_function, axis=1, arr=X)
    
    if propagation == 'fwd':
        y = data[:, -2]
    else:
        y = data[:, -1]

    if precision == '':
        result = [gpu_name, operator_name, propagation]
        print(f'Working on {gpu_name} {operator_name} {result[-1]}')
    else:
        result = [gpu_name, operator_name, precision, propagation]
        print(f'Working on {gpu_name} {operator_name} {precision} {result[-1]}')

    dtree_error, dtree_parameter = random_forest_tuning(X, y)
    xgboost_error, xgboost_parameter = xgboost_tuning(X, y)

    if dtree_error < xgboost_error:
        result.append('rforest')
        result.append(dtree_parameter)
        result.append(dtree_error)
    else:
        result.append('xgboost')
        result.append(xgboost_parameter)
        result.append(xgboost_error)         

    tools.write_one_result_to_csv(config_path, columns_name, np.array(result))




# def parse_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--gpu", default='NVIDIAA100-SXM4-80GB', type=str,
#                         help="GPU model name.")
#     args = parser.parse_args()
#     return args



# def main(args):
#     tuning(args.gpu)

def operator_tunning_all(file_folder, config_fold, gpu_name, operator_name_list, propagation_list, precision_list, columns_name):
    for operator_name in operator_name_list:
        for precision in precision_list:
            for propagation in propagation_list:
                operator_tunning(file_folder, config_fold, gpu_name, operator_name, propagation, precision, columns_name)



def operator_tunnning():
    file_folder = './Data/operators_A100'
    config_fold = './Model_config'
    gpu_name = "NVIDIAA100-SXM4-80GB"

    columns_name = ['GPU', 'Function', 'Precision', 'Propagation', 'Best_model', 'Best_config', 'Best_error%']

    # operator_tunning(file_folder, config_fold, gpu_name, 'firstStage_optimizer', 'dur', 'fp16', columns_name)  
    operator_tunning(file_folder, config_fold, gpu_name, 'middleStage_optimizer', 'dur', 'fp16', columns_name)  
    operator_tunning(file_folder, config_fold, gpu_name, 'lastStage_optimizer', 'dur', 'fp16', columns_name)  



def operator_tunnning_vista():
    # args = parse_arguments() 
    # main(args)
    file_folder = './Data/operators_GH200'
    config_fold = file_folder
    gpu_name = 'NVIDIAGH200120GB'
    # operator_name_list = ['baddbmm', 'bmm', 'embedding', 'fillmask', 'flash_atten', 'gelu', 'layernorm', 'linear_final', 'linear1', 'linear2', 'linear3', 'linear4', 'parallel_cross_entropy_128', 'res_add', 'RMSlayernorm', 'RoPE', 'ScaledUpperTriangMaskedSoftmax', 'softmax']

    # operator_name_list = ['ScaledUpperTriangMaskedSoftmax', 'softmax']

    operator_name_list = ['linear_final', 'linear1', 'linear2', 'linear3', 'linear4']


    precision_list = ['fp16']
    propagation_list = ['fwd', 'bwd'] # or bwd
    columns_name = ['GPU', 'Function', 'Precision', 'Propagation', 'Best_model', 'Best_config', 'Best_error%']

    operator_tunning_all(file_folder, config_fold, gpu_name, operator_name_list, propagation_list, precision_list, columns_name)

    # operator_tunning(file_folder, config_fold, gpu_name, 'firstStage_optimizer', 'dur', 'fp16', columns_name)  
    operator_tunning(file_folder, config_fold, gpu_name, 'middleStage_optimizer', 'dur', 'fp16', columns_name)  
    operator_tunning(file_folder, config_fold, gpu_name, 'lastStage_optimizer', 'dur', 'fp16', columns_name)  




if __name__ == '__main__':

    # operator_tunnning_vista()
    operator_tunnning()

    