import tools
import os
import numpy as np
import json


# get profiler json file name
def get_current_profiler_file_name(folder_path):
    filename_list = os.listdir(folder_path)
    for filename in filename_list:
        if os.path.splitext(filename)[-1] == '.json':
            return(filename)


def get_stages_profiler_path(target_path):
    path = target_path
    folders = sorted(
        [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))],
        key=lambda x: int(x) if x.isdigit() else x  # Sort numerically if the name is a number
    )
    path_list = []
    for folder in folders:
        temp_stage_folder = os.path.join(path, folder)
        temp_stage_profiler_path = os.path.join(temp_stage_folder, get_current_profiler_file_name(temp_stage_folder))
        path_list.append(temp_stage_profiler_path)
    return path_list


def get_dur_list_from_profiler(folder_path):
    target_folder = os.path.dirname(folder_path) + '/' + 'best_batch'
    stages_path_list = get_stages_profiler_path(target_folder)
    dur_list = []
    for path in stages_path_list:
        print(f'Loading: {path}')       
        with open(path, 'r') as f:
            temp_dur_list = json.load(f)['batch_dur_list']
            dur_list += temp_dur_list
    return dur_list
     

def get_the_iteration_statistic(profiler_list, model_names, save_path):
    columns_name = ['model', 'min', 'max', 'average', 'std', 'error of average to min']

    # save_name = '(' + tools.get_current_time() + ')' + 'iteration_statistics(s).csv'
    save_name = 'train_batch_statistics(s).csv'

    csv_save_path = save_path + '/' + save_name

    for i in range(len(model_names)):
        dur_list = []
        for folder_path in profiler_list[i]:
            dur_list = get_dur_list_from_profiler(folder_path)
        
        result = [
            model_names[i], 
            round(np.min(dur_list)/ 1000000, 2), 
            round(np.max(dur_list)/ 1000000, 2), 
            round(np.mean(dur_list)/ 1000000, 2), 
            round(np.std(dur_list)/ 1000000, 2),
            round(((np.mean(dur_list)-np.min(dur_list))/np.min(dur_list)) * 100, 4),
        ]

        tools.write_one_result_to_csv(csv_save_path, columns_name, result)

        print(f'Saving the result of {model_names[i]}.')  
        
        



if __name__ == "__main__":
    profiler_list = [
        ['./perlmutter/GPT_20B_4_4_8_32_4/31694544/PP4_MP4_DP8_ZERO1'],
        ['./perlmutter/GPT_20B_4_8_4_32_4/31697107/PP4_MP8_DP4_ZERO1'],
        ['./perlmutter/GPT_20B_8_4_4_32_4/31697099/PP8_MP4_DP4_ZERO1'],
        ['./perlmutter/llama_13B_4_8_2_16_4/31674706/PP4_MP8_DP2_ZERO1'],
        ['./perlmutter/llemma_7B_4_2_2_4_4/31664313/PP4_MP2_DP2_ZERO1'],
        ['./vista/GPT_20B_128_1/100696/PP4_MP4_DP8_ZERO1', './vista/GPT_20B_128_1/127321/PP4_MP4_DP8_ZERO1'],   # 2 better
        ['./vista/GPT_20B_128_1/100697/PP4_MP8_DP4_ZERO1', './vista/GPT_20B_128_1/127322/PP4_MP8_DP4_ZERO1'],   # 1 better
        ['./vista/GPT_20B_128_1/100698/PP8_MP4_DP4_ZERO1', './vista/GPT_20B_128_1/127323/PP8_MP4_DP4_ZERO1'],   # 1 better
        ['./vista/llama_13B_64_1/94078/PP4_MP8_DP2_ZERO1', './vista/llama_13B_64_1/136645/PP4_MP8_DP2_ZERO1'],  # 2 better
        ['./vista/llemma_7B_16_1/91703/PP4_MP2_DP2_ZERO1']
    ]

    model_names = [
        'GPT_20B_4_4_8_32_4_P',
        'GPT_20B_4_8_4_32_4_P',
        'GPT_20B_8_4_4_32_4_P',
        'llama_13B_4_8_2_16_4_P',
        'llemma_7B_4_2_2_4_4_P',
        'GPT_20B_4_4_8_32_4_V',
        'GPT_20B_4_8_4_32_4_V',
        'GPT_20B_8_4_4_32_4_V',
        'llama_13B_4_8_2_16_4_V',
        'llemma_7B_4_2_2_4_4_V'
    ]

    save_path = './statistics'


    get_the_iteration_statistic(profiler_list, model_names, save_path)