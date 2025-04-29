import pandas as pd
import numpy as np
import ast
import tools
import os
from pathlib import Path


def find_value(df, column_name, column_value, target_column_name):
    filtered_df = df[df[column_name] == column_value]
    if filtered_df.shape[0] == 0:
        return None
    else:
        target_index = filtered_df.index[0]
        value = df.iloc[target_index][target_column_name]
        if '[' in value:
            return_list = np.array(ast.literal_eval(value))
            return return_list[return_list != 0]
        else:
            return float(value)


def caculate_error(compare_type, df_prediction, df_measurement, prediction_module_name, measurement_module_name):
    search_column_name = 'module'
    measurement_target_column_name = 'gpu_runtime(us)'
    prediction_target_column_name = 'predition(us)'

    prediction_value = find_value(df_prediction, search_column_name, prediction_module_name, prediction_target_column_name)
    measurement_value = find_value(df_measurement, search_column_name, measurement_module_name, measurement_target_column_name)

    if compare_type == 'list':
        temp_measurement = measurement_value
        temp_prediction = prediction_value
        print(temp_measurement)
        print(temp_prediction)
        error = (temp_prediction - temp_measurement) / temp_measurement
        abs_mean_error = np.abs(error).mean()
    else:
        temp_measurement = np.sort(measurement_value)[-2] if type(measurement_value) is np.ndarray else measurement_value
        temp_prediction = np.sort(prediction_value)[-2] if type(prediction_value) is np.ndarray else prediction_value
        print(temp_measurement)
        print(temp_prediction)
        error = (temp_prediction - temp_measurement) / temp_measurement
        abs_mean_error = abs(error)

    return [prediction_module_name, temp_measurement, temp_prediction, error, abs_mean_error]


def caculate_error_average(compare_type, df_prediction, df_measurement, prediction_module_name, measurement_module_name):
    search_column_name = 'module'
    measurement_target_column_name = 'gpu_runtime(us)'
    prediction_target_column_name = 'predition(us)'

    prediction_value = find_value(df_prediction, search_column_name, prediction_module_name, prediction_target_column_name)
    measurement_value = find_value(df_measurement, search_column_name, measurement_module_name, measurement_target_column_name)

    if compare_type == 'list':
        temp_measurement = measurement_value
        temp_prediction = prediction_value
        print(temp_measurement)
        print(temp_prediction)
        error = (temp_prediction - temp_measurement) / temp_measurement
        abs_mean_error = np.abs(error).mean()

        temp_measurement = temp_measurement.tolist()
        temp_prediction = temp_prediction.tolist()
        error = error.tolist()
        abs_mean_error = abs_mean_error.tolist()
    else:
        temp_measurement = np.sort(measurement_value)[-2] if type(measurement_value) is np.ndarray else measurement_value
        temp_prediction = np.sort(prediction_value)[-2] if type(prediction_value) is np.ndarray else prediction_value
        print(temp_measurement)
        print(temp_prediction)
        error = (temp_prediction - temp_measurement) / temp_measurement
        abs_mean_error = abs(error)
    
     

    return [prediction_module_name, temp_measurement, temp_prediction, error, abs_mean_error]



def comparing_average(target_folder, prediction_file_name, measurement_file_name, compare_type_list, prediction_module_name_list, measurement_module_name_list):
    current_folder_name = os.path.basename(target_folder)
    parent_folder_name = Path(target_folder).parent.name

    time_string = tools.get_current_time()

    # save_result_file_name = '(' + time_string + ')' + parent_folder_name + '_' + current_folder_name + '_' + prediction_file_name.split('.')[0] + '_average' + '.csv'

    save_result_file_name = 'compares.csv'

    save_result_path = target_folder + '/' + save_result_file_name

    df_prediction = pd.read_csv(target_folder + '/' + prediction_file_name)
    df_measurement = pd.read_csv(target_folder + '/' + measurement_file_name)

    columns_name = ['module', 'measurement(us)', 'prediction(us)', 'error', 'abs_mean_error']

    for i in range(len(compare_type_list)):
        print(prediction_module_name_list[i])
        compare_type = compare_type_list[i]
        prediction_module_name = prediction_module_name_list[i]
        measurement_module_name = measurement_module_name_list[i]
        results = caculate_error_average(compare_type, df_prediction, df_measurement, prediction_module_name, measurement_module_name)
        # tools.write_one_result_to_csv(save_result_path, columns_name, np.array(results, dtype="object"))
        tools.write_one_result_to_csv(save_result_path, columns_name, results)




def get_compare_statistic_average(folder_path_list, prediction_file_name, measurement_file_name):

    compare_type_list = ['numeral', 'numeral', 'list', 'list', 'list', 'list', 'list', 'numeral', 'numeral', 'numeral']
    prediction_module_name_list = ['encoder_fwd', 'encoder_bwd', 'stage_fwd', 'stage_bwd', 'update', 'dp_allreduce', 'dp_allgather', 'pp_p2p', 'mp_allreduce', '1F1B_max_optimizer']
    measurement_module_name_list = ['encoder_fwd', 'encoder_bwd', 'stage_fwd', 'stage_bwd', 'optimizer', 'dp_allreduce', 'dp_allgather', 'pp_p2p', 'mp_allreduce', 'update']

    for folder_path in folder_path_list:
        print(folder_path)
        comparing_average(folder_path, prediction_file_name, measurement_file_name, compare_type_list, prediction_module_name_list, measurement_module_name_list)





if __name__ == '__main__':
    # get_compare_statistic()
    # add_update_to_measurements()
    # get_compare_statistic_average()

    folder_path_list = [
                        './perlmutter/GPT_20B_4_4_8_32_4/31694544/best_batch', 
                        './perlmutter/GPT_20B_4_8_4_32_4/31697107/best_batch',
                        './perlmutter/GPT_20B_8_4_4_32_4/31697099/best_batch', 
                        './perlmutter/llama_13B_4_8_2_16_4/31674706/best_batch',   # cannot find dp_allgather
                        './perlmutter/llemma_7B_4_2_2_4_4/31664313/best_batch', #  can find dp_allgather
                        './vista/GPT_20B_128_1/100697/best_batch', 
                        './vista/GPT_20B_128_1/100698/best_batch',
                        './vista/GPT_20B_128_1/127321/best_batch', 
                        # './vista/GPT_20B_128_1/127322/best_batch', 
                        # './vista/GPT_20B_128_1/127323/best_batch',
                        # './vista/llama_13B_64_1/94078/best_batch',  #  cannot find dp_allgather
                        './vista/llama_13B_64_1/136645/best_batch',   #  cannot find dp_allgather
                        './vista/llemma_7B_16_1/91703/best_batch'   #  cannot find dp_allgather
                        ]

    prediction_file_name = 'predicts.csv'
    measurement_file_name = 'module_times_average.csv'

    get_compare_statistic_average(folder_path_list, prediction_file_name, measurement_file_name)




