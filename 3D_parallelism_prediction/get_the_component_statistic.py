import pandas as pd
import tools
import os
import numpy as np
import json
import ast


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


def get_df_list(profiler_list, csv_name):
    df_list = []
    for path in profiler_list:
        df = pd.read_csv(path + '/' + csv_name)
        df_list.append(df)
    return df_list


def extract_errors(df_list, model_names, save_path):
    # components = [
    #     'encoder_fwd',
    #     'encoder_bwd',
    #     'stage_fwd',
    #     'stage_bwd',
    #     'dp_allreduce',
    #     'dp_allgather',
    #     'update',
    #     'mp_allreduce',
    #     'pp_p2p',
    #     'all_F_all_B_max_optimizer'
    # ]

    columns_name = ['componet'] + model_names
    path = save_path + '/components_statistic.csv'

    # for 'encoder_fwd'
    encoder_fwd_errors = ['encoder_fwd']
    for df in df_list:
        encoder_fwd_errors.append(round(find_value(df, 'module', 'encoder_fwd', 'error')*100, 2))
    print(encoder_fwd_errors)
    tools.write_one_result_to_csv(path, columns_name, encoder_fwd_errors)

    # for 'encoder_bwd'
    encoder_bwd_errors = ['encoder_bwd']
    for df in df_list:
        encoder_bwd_errors.append(round(find_value(df, 'module', 'encoder_bwd', 'error')*100, 2))
    print(encoder_bwd_errors)
    tools.write_one_result_to_csv(path, columns_name, encoder_bwd_errors)

    # for 'stage_fwd'
    stage_fwd_errors = ['stage_fwd_max']
    for df in df_list:
        prediction_list = find_value(df, 'module', 'stage_fwd', 'prediction(us)')
        max_index = np.argmax(prediction_list)
        error_list = find_value(df, 'module', 'stage_fwd', 'error')
        stage_fwd_errors.append(round(error_list[max_index]*100, 2))
    print(stage_fwd_errors)
    tools.write_one_result_to_csv(path, columns_name, stage_fwd_errors)

    # for 'stage_bwd'
    stage_bwd_errors = ['stage_bwd_max']
    for df in df_list:
        prediction_list = find_value(df, 'module', 'stage_bwd', 'prediction(us)')
        max_index = np.argmax(prediction_list)
        error_list = find_value(df, 'module', 'stage_bwd', 'error')
        stage_bwd_errors.append(round(error_list[max_index]*100, 2))
    print(stage_bwd_errors)
    tools.write_one_result_to_csv(path, columns_name, stage_bwd_errors)

    # for 'dp_allreduce'
    dp_allreduce_errors = ['dp_allreduce(1st stage)']
    for df in df_list:
        error_list = find_value(df, 'module', 'dp_allreduce', 'error')
        dp_allreduce_errors.append(round(error_list[0]*100, 2))
    print(dp_allreduce_errors)
    tools.write_one_result_to_csv(path, columns_name, dp_allreduce_errors)


    # for 'dp_allgather' and 'update'
    dp_allgather_errors = ['dp_allgather(max_update)']
    update_errors = ['max_update']
    for df in df_list:
        prediction_list = find_value(df, 'module', 'update', 'prediction(us)')
        max_index = np.argmax(prediction_list)

        dp_allgather_error_list = find_value(df, 'module', 'dp_allgather', 'error')
        dp_allgather_errors.append(round(dp_allgather_error_list[max_index]*100, 2))

        update_error_list = find_value(df, 'module', 'update', 'error')
        update_errors.append(round(update_error_list[max_index]*100, 2))
    print(dp_allgather_errors)
    print(update_errors)
    tools.write_one_result_to_csv(path, columns_name, dp_allgather_errors)
    tools.write_one_result_to_csv(path, columns_name, update_errors)


    # for 'mp_allreduce' 
    mp_allreduce_errors = ['mp_allreduce']
    for df in df_list:
        error = find_value(df, 'module', 'mp_allreduce', 'error')
        mp_allreduce_errors.append(round(error*100, 2))
    print(mp_allreduce_errors)
    tools.write_one_result_to_csv(path, columns_name, mp_allreduce_errors)


    # for 'pp_p2p' 
    pp_p2p_errors = ['pp_p2p']
    for df in df_list:
        error = find_value(df, 'module', 'pp_p2p', 'error')
        pp_p2p_errors.append(round(error*100, 2))
    print(pp_p2p_errors)
    tools.write_one_result_to_csv(path, columns_name, pp_p2p_errors)


    # for 'Overall' 
    Overall_errors = ['Overall']
    for df in df_list:
        error = find_value(df, 'module', '1F1B_max_optimizer', 'error')
        Overall_errors.append(round(error*100, 2))
    print(Overall_errors)
    tools.write_one_result_to_csv(path, columns_name, Overall_errors)


def main():
    profiler_list = [
        './perlmutter/GPT_20B_4_4_8_32_4/31694544/best_batch',
        './vista/GPT_20B_128_1/127321/best_batch',   # 2 better
        './perlmutter/GPT_20B_4_8_4_32_4/31697107/best_batch',
        './vista/GPT_20B_128_1/100697/best_batch',   # 1 better
        './perlmutter/GPT_20B_8_4_4_32_4/31697099/best_batch',
        './vista/GPT_20B_128_1/100698/best_batch',   # 1 better
        './perlmutter/llama_13B_4_8_2_16_4/31674706/best_batch',
        './vista/llama_13B_64_1/136645/best_batch',  # 2 better
        './perlmutter/llemma_7B_4_2_2_4_4/31664313/best_batch',
        './vista/llemma_7B_16_1/91703/best_batch'
    ]

    model_names = [
        'GPT_20B_4_4_8_P',
        'GPT_20B_4_4_8_V',
        'GPT_20B_4_8_4_P',
        'GPT_20B_4_8_4_V',
        'GPT_20B_8_4_4_P',
        'GPT_20B_8_4_4_V',
        'llama_13B_4_8_2_P',
        'llama_13B_4_8_2_V',
        'llemma_7B_4_2_2_P',
        'llemma_7B_4_2_2_V'
    ]

    save_path = './statistics'

    components = [
        'encoder_fwd',
        'encoder_bwd',
        'stage_fwd',
        'stage_bwd',
        'dp_allreduce',
        'dp_allgather',
        'update',
        'mp_allreduce',
        'pp_p2p',
        'all_F_all_B_max_optimizer'
    ]

    df_list = get_df_list(profiler_list, 'compares.csv')

    extract_errors(df_list, model_names, save_path)


if __name__ == "__main__":
    main()