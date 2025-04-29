import pandas as pd
import numpy as np
import ast
import tools
import os
from pathlib import Path
import matplotlib.pyplot as plt


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


def caculate_protion_max_optimizer(prediction_file_folder, prediction_file_name, encoder_layers_list, mp_snycs, iters_per_update, pp_stages):
    prediction_file_path = prediction_file_folder + '/' + prediction_file_name

    time_string = tools.get_current_time()

    save_path = prediction_file_folder +  '/' + '(' + time_string + ')' + prediction_file_name.split('.')[0] + '_portion.csv'

    df = pd.read_csv(prediction_file_path)

    columns_name = ['module', 'portion']

    total_runtime = find_value(df, 'module', 'all_F_all_B_max_optimizer', 'predition(us)')

    optimizer = find_value(df, 'module', 'optimizer', 'predition(us)')

    stage_fwd = find_value(df, 'module', 'stage_fwd', 'predition(us)')
    stage_bwd = find_value(df, 'module', 'stage_bwd', 'predition(us)')

    stage_fwd_max_index = np.argmax(stage_fwd) 
    stage_bwd_max_index = np.argmax(stage_bwd) 

    number_of_encoders_fwd = encoder_layers_list[stage_fwd_max_index]
    number_of_encoders_bwd = encoder_layers_list[stage_bwd_max_index]
    
    multiply = (iters_per_update + pp_stages -1)

    encoder_fwd = find_value(df, 'module', 'encoder_fwd', 'predition(us)')
    encoder_fwd_total = encoder_fwd * (number_of_encoders_fwd + number_of_encoders_bwd) * multiply
    results = ['encoder_fwd', encoder_fwd_total / total_runtime]
    tools.write_one_result_to_csv(save_path, columns_name, np.array(results, dtype="object"))

    encoder_bwd = find_value(df, 'module', 'encoder_bwd', 'predition(us)')
    encoder_bwd_total = encoder_bwd * number_of_encoders_bwd * multiply
    results = ['encoder_bwd', encoder_bwd_total / total_runtime]
    tools.write_one_result_to_csv(save_path, columns_name, np.array(results, dtype="object"))

    stage_fwd_total = np.max(stage_fwd) * multiply
    results = ['stage_fwd', stage_fwd_total / total_runtime]
    tools.write_one_result_to_csv(save_path, columns_name, np.array(results, dtype="object"))

    stage_bwd_total = np.max(stage_bwd) * multiply
    results = ['stage_bwd', stage_bwd_total / total_runtime]
    tools.write_one_result_to_csv(save_path, columns_name, np.array(results, dtype="object"))

    dp_allreduce = find_value(df, 'module', 'dp_allreduce', 'predition(us)')[0]
    results = ['dp_allreduce', dp_allreduce / total_runtime]
    tools.write_one_result_to_csv(save_path, columns_name, np.array(results, dtype="object"))

    dp_allgather = find_value(df, 'module', 'dp_allgather', 'predition(us)')[np.argmax(optimizer)]
    results = ['dp_allgather', dp_allgather / total_runtime]
    tools.write_one_result_to_csv(save_path, columns_name, np.array(results, dtype="object"))

    results = ['optimizer', np.max(optimizer) / total_runtime]
    tools.write_one_result_to_csv(save_path, columns_name, np.array(results, dtype="object"))

    mp_allreduce = find_value(df, 'module', 'mp_allreduce', 'predition(us)')
    mp_allreduce_total = (mp_allreduce * mp_snycs[0] * number_of_encoders_fwd + mp_allreduce * (mp_snycs[0] + mp_snycs[1]) * number_of_encoders_bwd) * multiply
    results = ['mp_allreduce', mp_allreduce_total / total_runtime]
    tools.write_one_result_to_csv(save_path, columns_name, np.array(results, dtype="object"))

    pp_p2p = find_value(df, 'module', 'mp_allreduce', 'predition(us)')
    pp_p2p_fwd = pp_p2p * multiply if stage_fwd_max_index != 2 else 0 
    pp_p2p_bwd = pp_p2p * multiply if stage_bwd_max_index != 0 else 0 
    pp_p2p_total = pp_p2p_fwd + pp_p2p_bwd
    results = ['pp_p2p', pp_p2p_total / total_runtime]
    tools.write_one_result_to_csv(save_path, columns_name, np.array(results, dtype="object"))


def caculate_protion_max_optimizer_return_list(prediction_file_folder, prediction_file_name, encoder_layers_list, mp_snycs, iters_per_update, pp_stages):
    prediction_file_path = prediction_file_folder + '/' + prediction_file_name

    # time_string = tools.get_current_time()

    # save_path = prediction_file_folder +  '/' + '(' + time_string + ')' + prediction_file_name.split('.')[0] + '_portion.csv'

    df = pd.read_csv(prediction_file_path)

    columns_name = ['module', 'portion']

    total_runtime = find_value(df, 'module', '1F1B_max_optimizer', 'predition(us)')

    optimizer = find_value(df, 'module', 'optimizer', 'predition(us)')

    stage_fwd = find_value(df, 'module', 'stage_fwd', 'predition(us)')
    stage_bwd = find_value(df, 'module', 'stage_bwd', 'predition(us)')

    stage_fwd_max_index = np.argmax(stage_fwd) 
    stage_bwd_max_index = np.argmax(stage_bwd) 

    number_of_encoders_fwd = encoder_layers_list[stage_fwd_max_index]
    number_of_encoders_bwd = encoder_layers_list[stage_bwd_max_index]
    
    multiply = (iters_per_update + pp_stages -1)

    results_list = []

    encoder_fwd = find_value(df, 'module', 'encoder_fwd', 'predition(us)')
    encoder_fwd_total = encoder_fwd * (number_of_encoders_fwd + number_of_encoders_bwd) * multiply
    results_list.append(encoder_fwd_total / total_runtime)

    encoder_bwd = find_value(df, 'module', 'encoder_bwd', 'predition(us)')
    encoder_bwd_total = encoder_bwd * number_of_encoders_bwd * multiply
    results_list.append(encoder_bwd_total / total_runtime)

    stage_fwd_total = np.max(stage_fwd) * multiply
    results_list.append(stage_fwd_total / total_runtime)

    stage_bwd_total = np.max(stage_bwd) * multiply
    results_list.append(stage_bwd_total / total_runtime)

    dp_allreduce = find_value(df, 'module', 'dp_allreduce', 'predition(us)')[0]
    results_list.append(dp_allreduce / total_runtime)

    update_list = find_value(df, 'module', 'dp_allgather', 'predition(us)') + optimizer

    dp_allgather = find_value(df, 'module', 'dp_allgather', 'predition(us)')[np.argmax(update_list)]
    results_list.append(dp_allgather / total_runtime)

    # append update
    results_list.append(np.max(update_list)  / total_runtime)

    mp_allreduce = find_value(df, 'module', 'mp_allreduce', 'predition(us)')
    mp_allreduce_total = (mp_allreduce * mp_snycs[0] * number_of_encoders_fwd + mp_allreduce * (mp_snycs[0] + mp_snycs[1]) * number_of_encoders_bwd) * multiply
    results_list.append(mp_allreduce_total  / total_runtime)

    pp_p2p = find_value(df, 'module', 'mp_allreduce', 'predition(us)')
    pp_p2p_fwd = pp_p2p * multiply if stage_fwd_max_index != 2 else 0 
    pp_p2p_bwd = pp_p2p * multiply if stage_bwd_max_index != 0 else 0 
    pp_p2p_total = pp_p2p_fwd + pp_p2p_bwd
    results_list.append(pp_p2p_total  / total_runtime)

    return results_list



def gpt20b_4_4_8(prediction_file_name):
    prediction_file_folder_list = ['./perlmutter/GPT_20B_4_4_8_32_4/31694544', 
                             './vista/GPT_20B_128_1/100696']
    encoder_layers_list = [11, 12, 9]
    mp_snycs = [1, 2]
    iters_per_update = 16
    pp_stages = 4
    for prediction_file_folder in prediction_file_folder_list:
        caculate_protion_max_optimizer(prediction_file_folder, prediction_file_name, encoder_layers_list, mp_snycs, iters_per_update, pp_stages)


def gpt20b_4_8_4(prediction_file_name):
    prediction_file_folder_list = ['./perlmutter/GPT_20B_4_8_4_32_4/31697107', 
                             './vista/GPT_20B_128_1/100697']
    encoder_layers_list = [11, 12, 9]
    mp_snycs = [1, 2]
    iters_per_update = 16
    pp_stages = 4
    for prediction_file_folder in prediction_file_folder_list:
        caculate_protion_max_optimizer(prediction_file_folder, prediction_file_name, encoder_layers_list, mp_snycs, iters_per_update, pp_stages)


def gpt20b_8_4_4(prediction_file_name):
    prediction_file_folder_list = ['./perlmutter/GPT_20B_8_4_4_32_4/31697099', 
                             './vista/GPT_20B_128_1/100698']
    encoder_layers_list = [5, 6, 3]
    mp_snycs = [1, 2]
    iters_per_update = 16
    pp_stages = 8
    for prediction_file_folder in prediction_file_folder_list:
        caculate_protion_max_optimizer(prediction_file_folder, prediction_file_name, encoder_layers_list, mp_snycs, iters_per_update, pp_stages)


def llama_13B_4_8_2():
    prediction_file_folder_list = ['./perlmutter/llama_13B_4_8_2_16_4/31674706', 
                             './vista/llama_13B_64_1/94078']
    encoder_layers_list = [10, 11, 8]
    mp_snycs = [2, 2]
    iters_per_update = 16
    pp_stages = 4
    for prediction_file_folder in prediction_file_folder_list:
        caculate_protion_max_optimizer(prediction_file_folder, prediction_file_name, encoder_layers_list, mp_snycs, iters_per_update, pp_stages)


def llemma_7B_4_2_2():
    prediction_file_folder_list = ['./perlmutter/llemma_7B_4_2_2_4_4/31664313', 
                             './vista/llemma_7B_16_1/91703']
    encoder_layers_list = [8, 9, 6]
    mp_snycs = [2, 2]
    iters_per_update = 8
    pp_stages = 4
    for prediction_file_folder in prediction_file_folder_list:
        caculate_protion_max_optimizer(prediction_file_folder, prediction_file_name, encoder_layers_list, mp_snycs, iters_per_update, pp_stages)




def draw_bar_chart(categories, values_list, sub_chart_title_list):
    # categories = ['A', 'B', 'C', 'D']
    # values_list = [np.random.randint(5, 25, size=len(categories)) for _ in range(num_charts)]

    # empty_categories = [' ' for i in range(len(categories))]

    num_charts = 10

    # Create figure and axes
    fig, axes = plt.subplots(2, 5, figsize=(15, 10))
    # fig.suptitle("10 Horizontal Bar Charts in a 2x5 Grid")

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Generate 10 bar charts
    for i in range(num_charts):
        values = values_list[i]
        values = [int(a *10000) / 100 for a in values[::-1]]

        bars = axes[i].barh(categories[::-1], values, color='skyblue')
        axes[i].set_title(sub_chart_title_list[i], fontsize=15, fontweight='bold')
        
        if i >= 5:
            axes[i].set_xlabel('Percentage(%)', fontsize=15)

        # Remove y-axis category labels
        if i % 5 > 0:
            axes[i].set_yticklabels([])

        axes[i].tick_params(axis='both', labelsize=15)

        max_value = max(values)
        # Extend x-axis slightly beyond the longest bar
        axes[i].set_xlim(0, max_value * 1.4)

        # Add value labels at the end of each bar
        for bar in axes[i].containers[0]:
            axes[i].text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                        f'{bar.get_width()}', va='center', ha='left', fontsize=15)


    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_folder = './bar_charts'

    save_path = save_folder + '/' + '(' + tools.get_current_time() + ')' + 'horizontal_bar_charts_grid.png'

    plt.savefig(save_path, dpi=300, bbox_inches='tight')




def get_bar_chart(prediction_file_name):

    prediction_file_folder_list = [
        './perlmutter/GPT_20B_4_4_8_32_4/31694544/best_batch',
        './perlmutter/GPT_20B_4_8_4_32_4/31697107/best_batch',
        './perlmutter/GPT_20B_8_4_4_32_4/31697099/best_batch',
        './perlmutter/llama_13B_4_8_2_16_4/31674706/best_batch',
        './perlmutter/llemma_7B_4_2_2_4_4/31664313/best_batch',
        './vista/GPT_20B_128_1/127321/best_batch',
        './vista/GPT_20B_128_1/100697/best_batch',
        './vista/GPT_20B_128_1/100698/best_batch',
        './vista/llama_13B_64_1/136645/best_batch',
        './vista/llemma_7B_16_1/91703/best_batch'
    ]

    config_list = [
        [[11, 12, 9], [1, 2], 16, 4],
        [[11, 12, 9], [1, 2], 16, 4],
        [[5, 6, 3], [1, 2], 16, 8],
        [[10, 11, 8], [2, 2], 16, 4],
        [[8, 9, 6], [2, 2], 8, 4],
        [[11, 12, 9], [1, 2], 16, 4],
        [[11, 12, 9], [1, 2], 16, 4],
        [[5, 6, 3], [1, 2], 16, 8],
        [[10, 11, 8], [2, 2], 16, 4],
        [[8, 9, 6], [2, 2], 8, 4]
    ]

    sub_chart_title_list = [
        'GPT-20B(4-4-8)-P',
        'GPT-20B(4-8-4)-P',
        'GPT-20B(8-4-4)-P',
        'LLaMA-13B(4-8-2)-P',
        'Llemma-7B(4-2-2)-P',
        'GPT-20B(4-4-8)-V',
        'GPT-20B(4-8-4)-V',
        'GPT-20B(8-4-4)-V',
        'LLaMA-13B(4-8-2)-V',
        'Llemma-7B(4-2-2)-V'
    ]

    results_list = []

    for i in range(len(prediction_file_folder_list)):
        prediction_file_folder = prediction_file_folder_list[i]
        encoder_layers_list, mp_snycs, iters_per_update, pp_stages = config_list[i]
        results = caculate_protion_max_optimizer_return_list(prediction_file_folder, prediction_file_name, encoder_layers_list, mp_snycs, iters_per_update, pp_stages)
        results_list.append(results)


    categories = ['encoder_fwd', 'encoder_bwd', 'stage_fwd', 'stage_bwd', 'dp_allreduce', 'dp_allgather', 'update', 'mp_allreduce', 'pp_p2p']
    draw_bar_chart(categories, results_list, sub_chart_title_list)
    



if __name__ == '__main__':
    prediction_file_name = 'predicts.csv'
    get_bar_chart(prediction_file_name)




