import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

import ast

import tools
import os
import importlib
import math


MODEL_DICT = {}
NCCL_DICT = {}


def rebuild_rforest(best_params, X_train, y_train):
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    return model


def rebuild_xgboost(best_params, X_train, y_train):
    model = xgb.XGBRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    return model


def rebuild_model(gpu_name, model_name, model_dict, config_folder, data_folder):
    # GPU_Name = torch.cuda.get_device_name(0).replace(' ', '')
    config_path = config_folder + '/' + gpu_name + '.csv'
    
    df = pd.read_csv(config_path)
    df['Best_config'] = df['Best_config'].apply(ast.literal_eval)
    

    precision_str, propagation_str = model_name.split('_')[-2:]
    function_str = '_'.join(model_name.split('_')[0:-2])

    target_i = df[(df['Function'] == function_str) & (df['Precision'] == precision_str) & (df['Propagation'] == propagation_str)].index[0]

    model_type = df.iloc[target_i]['Best_model']
    best_params = df.iloc[target_i]['Best_config']

    data_path = data_folder + '/' + gpu_name + '_' + function_str + '_' + precision_str + '.csv'
    data = pd.read_csv(data_path).to_numpy()

    if propagation_str == 'dur':
        X_train = data[:, 0:-1]
    else:    
        X_train = data[:, 0:-2]

    map_function = get_map_function('layer_input_to_predictor_input', function_str)
    X_train = np.apply_along_axis(map_function, axis=1, arr=X_train)
    
    if propagation_str == 'fwd':
        y_train = data[:, -2]
    else:
        y_train = data[:, -1]

    if model_type == 'xgboost':
        model = rebuild_xgboost(best_params, X_train, y_train)
    else:
        model = rebuild_rforest(best_params, X_train, y_train)

    model_dict[model_name] = model

    return model


def predict_operator(gpu_name, function, precision, propagation, shape, model_dict, config_folder, data_folder):
    model_name = function + '_' + precision + '_' + propagation
    if model_name in model_dict:
        model = model_dict[model_name]
    else:
        model = rebuild_model(gpu_name, model_name, model_dict, config_folder, data_folder)

    map_function = get_map_function('layer_input_to_predictor_input', function)
    shape_new = map_function(shape)
    
    predict = model.predict([shape_new])

    print(f'Function:{model_name}\t Input:{shape}\t PredictorInput:{shape_new}\t Prediction:{predict[0]}')

    return predict[0]


def get_layer_input_shape(encoder_config, module_name, function_name):
    module = importlib.import_module(module_name)
    return getattr(module, function_name)(encoder_config)


def get_map_function(module_name, function_name):
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def operator_statistic(data_folder, config_folder, gpu_name, function_list, precision, encoder_config, propagation):
    time_cost = 0

    for function in function_list:
        module_name = 'encoder_config_to_layer_input'
        shape = get_layer_input_shape(encoder_config, module_name, function)
        result = predict_operator(gpu_name, function, precision, propagation, shape, MODEL_DICT, config_folder, data_folder)
        time_cost += result
        
    return time_cost



if __name__ == '__main__':
    shape = 20971520
    mp = 2
    gpu_per_node = 4
    timecost = mp_allreduce(shape, mp, gpu_per_node)
    print(timecost)