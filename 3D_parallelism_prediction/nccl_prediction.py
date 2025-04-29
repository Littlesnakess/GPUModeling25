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
    
    features = model_name.split('_')
    precision = features[-3]
    nodes = features[-2]
    gpus_per_node = features[-1]
    operator_name = '_'.join(features[0:-3])
    column_indexes = [0]

    data_path = data_folder + '/' + gpu_name + '_' + model_name + '.csv'
    if not os.path.isfile(data_path):
        data_path = data_folder + '/' + gpu_name + '_' + operator_name + '_' + precision + '.csv'
        column_indexes = [0, 1, 2]
        dict_key_name = operator_name + '_' + precision
        target_i = df[(df['Function'] == operator_name) & (df['Precision'] == precision) & (df['Nodes'] == 0) & (df['GPUsPerNode'] == 0)].index[0]
    else:
        dict_key_name = model_name
        target_i = df[(df['Function'] == operator_name) & (df['Precision'] == precision) & (df['Nodes'] == int(nodes)) & (df['GPUsPerNode'] == int(gpus_per_node))].index[0]


    model_type = df.iloc[target_i]['Best_model']
    best_params = df.iloc[target_i]['Best_config']

    data = pd.read_csv(data_path).to_numpy()
    
    X_train = data[:, column_indexes]
    y_train = data[:, -1]
    print(f'Rebuilding {data_path}')
    
    if model_type == 'xgboost':
        model = rebuild_xgboost(best_params, X_train, y_train)
    else:
        model = rebuild_rforest(best_params, X_train, y_train)

    model_dict[dict_key_name] = model

    return model


def predict_nccl_operator(gpu_name, model_name, shape, model_dict, config_folder, data_folder):
    data_path = data_folder + '/' + gpu_name + '_' + model_name + '.csv'
    if os.path.isfile(data_path):
        dict_key_name = model_name
    else:
        dict_key_name = '_'.join(model_name.split('_')[0:-2])

    if dict_key_name in model_dict:
        model = model_dict[dict_key_name]
    else:
        model = rebuild_model(gpu_name, model_name, model_dict, config_folder, data_folder)
    
    if os.path.isfile(data_path):
        predict_input = [[shape]]
        predict = model.predict(predict_input)
    else:
        features = model_name.split('_')
        nodes = features[-2]
        gpus_per_node = features[-1]
        predict_input = [[shape, int(nodes), int(gpus_per_node)]]
        predict = model.predict(predict_input)

    # print(f'Function:{dict_key_name}\t Input:{predict_input}\t Prediction:{predict[0]}')
    print(f'Function:{model_name}\t Input:{shape}\t Prediction:{predict[0]}')

    return predict[0]


def mp_allreduce(nccl_data_folder, nccl_config_folder, gpu_name, shape, mp, gpu_per_node):
    nccl_function = 'allreduce_fp16'
    mp_gpu_per_node = mp if mp // gpu_per_node == 0 else gpu_per_node
    nodes = 1 if mp // mp_gpu_per_node == 0 else mp // mp_gpu_per_node
    nccl_function = '_'.join([nccl_function, str(nodes), str(mp_gpu_per_node)])

    print(nccl_function)

    timecost = predict_nccl_operator(gpu_name, nccl_function, shape, NCCL_DICT, nccl_config_folder, nccl_data_folder)
    return timecost


def dp_allreduce(nccl_data_folder, nccl_config_folder, gpu_name, shape, mp, dp, gpu_per_node):
    nccl_function = 'allreduce_large_fp16'
    dp_gpu_per_node = 1 if gpu_per_node // mp ==0 else gpu_per_node // mp
    nodes = 1 if dp // dp_gpu_per_node == 0 else dp // dp_gpu_per_node
    nccl_function = '_'.join([nccl_function, str(nodes), str(dp_gpu_per_node)])
    timecost = predict_nccl_operator(gpu_name, nccl_function, shape, NCCL_DICT, nccl_config_folder, nccl_data_folder)
    return timecost


def dp_allgather(nccl_data_folder, nccl_config_folder, gpu_name, shape, mp, dp, gpu_per_node):
    nccl_function = 'allgather_large_fp16'
    dp_gpu_per_node = 1 if gpu_per_node // mp ==0 else gpu_per_node // mp
    nodes = 1 if dp // dp_gpu_per_node == 0 else dp // dp_gpu_per_node
    nccl_function = '_'.join([nccl_function, str(nodes), str(dp_gpu_per_node)])
    timecost = predict_nccl_operator(gpu_name, nccl_function, shape, NCCL_DICT, nccl_config_folder, nccl_data_folder)
    return timecost


def pp_p2p(nccl_data_folder, nccl_config_folder, gpu_name, shape, mp, dp, pp, gpu_per_node):
    nccl_function = 'p2p_fp16'
    if mp*dp*pp >= gpu_per_node:
        nodes = 2
        pp_gpu_per_node = 1
    else:
        nodes = 1
        pp_gpu_per_node = 2
    nccl_function = '_'.join([nccl_function, str(nodes), str(pp_gpu_per_node)])
    timecost = predict_nccl_operator(gpu_name, nccl_function, shape//mp, NCCL_DICT, nccl_config_folder, nccl_data_folder)
    return timecost


def mp_allreduce_test():
    nccl_data_folder = 'Data/nccl_perlmutter/required'
    nccl_config_folder = nccl_data_folder
    gpu_name = 'NVIDIAA100-SXM4-40GB'
    shape = 20971520
    mp_gpu_list = [[2, 2], [4, 4], [2, 1], [4, 2], [8, 4], [16, 4]]
    for mp_gpu in mp_gpu_list:
        mp, gpu_per_node= mp_gpu
        timecost = mp_allreduce(nccl_data_folder, nccl_config_folder, gpu_name, shape, mp, gpu_per_node)


def dp_allreduce_test():
    nccl_data_folder = 'Data/nccl_perlmutter/required'
    nccl_config_folder = nccl_data_folder
    gpu_name = 'NVIDIAA100-SXM4-40GB'
    shape = 134217728
    mp_dp_gpu_list = [[1, 2, 2], [1, 4, 4], [1, 2, 1], [1, 4, 2], [1, 8, 4], [1, 4, 1], [1, 8, 1], [1, 16, 1]]
    for mp_gpu in mp_dp_gpu_list:
        mp, dp, gpu_per_node= mp_gpu
        timecost = dp_allreduce(nccl_data_folder, nccl_config_folder, gpu_name, shape, mp, dp, gpu_per_node)


def dp_allgather_test():
    nccl_data_folder = 'Data/nccl_perlmutter/required'
    nccl_config_folder = nccl_data_folder
    gpu_name = 'NVIDIAA100-SXM4-40GB'
    shape = 134217728
    mp_dp_gpu_list = [[1, 2, 2], [1, 4, 4], [1, 2, 1], [1, 4, 2], [1, 8, 4], [1, 4, 1], [1, 8, 1], [1, 16, 1]]
    for mp_gpu in mp_dp_gpu_list:
        mp, dp, gpu_per_node= mp_gpu
        timecost = dp_allgather(nccl_data_folder, nccl_config_folder, gpu_name, shape, mp, dp, gpu_per_node)

        
def pp_p2p_test():
    nccl_data_folder = 'Data/nccl_perlmutter/required'
    nccl_config_folder = nccl_data_folder
    gpu_name = 'NVIDIAA100-SXM4-40GB'
    shape = 20971520
    mp_dp_pp_gpu_list = [[1, 1, 2, 4], [2, 2, 2, 4]]
    for mp_gpu in mp_dp_pp_gpu_list:
        mp, dp, pp, gpu_per_node= mp_gpu
        timecost = pp_p2p(nccl_data_folder, nccl_config_folder, gpu_name, shape, mp, dp, pp, gpu_per_node)


if __name__ == '__main__':
    # mp_allreduce_test()
    # dp_allreduce_test()
    # dp_allgather_test()
    pp_p2p_test()