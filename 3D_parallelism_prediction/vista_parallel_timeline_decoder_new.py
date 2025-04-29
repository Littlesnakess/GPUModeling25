import json
# import ujson as json
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import time
import os
import numpy as np
import csv
import tools

import argparse
import tools

# DATA = None
# CPU_OP_ITEMS = None
# CUDA_RUNTIME_ITEMS = None
# KERNEL_ITEMS = None
# NUM_THREADS = 4


from pathlib import Path


class TimelineDecoder:
    def __init__(self, data_path, num_threads):
        self.data = None
        self.cpu_op_items = None
        self.cuda_runtime_items = None
        self.kernel_items = None
        self.num_threads = num_threads

        with open(data_path, 'r') as f:
            data = json.load(f)

        self.data = data["traceEvents"] 
        self.cpu_op_items = self.get_cpu_op_items(self.data)
        self.cuda_runtime_items = self.get_cuda_runtime_items(self.data)
        self.kernel_items = self.get_kernel_items(self.data)



    def get_nested_value(self, data, key_path):
        for key in key_path:
            if isinstance(data, list):
                # If data is a list, apply the rest of the path to each item in the list
                return [get_nested_value(item, key_path[key_path.index(key) + 1:]) for item in data]
            data = data.get(key, {})
        return data


    def search_item_nested_range_with_list(self, json_data, key_path, min_value, max_value):
        results = []
        for item in json_data:
            values = self.get_nested_value(item, key_path)

            if isinstance(values, list) and any(min_value < v < max_value for v in values if isinstance(v, (int, float))):
                results.append(item)
            elif isinstance(values, (int, float)) and min_value < values < max_value:
                results.append(item)
        return results


    def search_item_value_with_list(self, json_data, key_path, target_values):
        results = []
        for item in json_data:
            values = self.get_nested_value(item, key_path)

            if isinstance(values, list) and any(v in target_values for v in values if isinstance(v, (int, float, str))):
                results.append(item)
            elif isinstance(values, (int, float, str)) and values in target_values:
                results.append(item)
        return results



    def search_in_range_cpu_ops(self, json_data, target_value):
        results = []
        for item in json_data:
            value = item.get('ts') + item.get('dur')
            if value > target_value:
                results.append(item)
        return results


    def chunk_data(self, data, num_chunks):
        chunk_size = len(data) // num_chunks + 1
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


    def parallel_search_nested_range(self, data, key_path, min_value, max_value, num_threads=4):

        data_chunks = self.chunk_data(data, num_threads)
        
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self.search_item_nested_range_with_list, chunk, key_path, min_value, max_value)
                for chunk in data_chunks
            ]
            for future in as_completed(futures):
                results.extend(future.result())
        
        return results


    def parallel_search_values(self, data, key_path, target_values, num_threads=4):

        data_chunks = self.chunk_data(data, num_threads)
        
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self.search_item_value_with_list, chunk, key_path, target_values)
                for chunk in data_chunks
            ]
            for future in as_completed(futures):
                results.extend(future.result())
        return results


    def parallel_search_in_range_cpu_ops(self, data, target_value, num_threads=4):

        data_chunks = self.chunk_data(data, num_threads)
        
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self.search_in_range_cpu_ops, chunk, target_value)
                for chunk in data_chunks
            ]
            for future in as_completed(futures):
                results.extend(future.result())
        return sorted(results, key=lambda x: x.get('dur'))


    def get_cpu_op_items(self, data):
        key_path = ['cat']
        target_values = ['cpu_op', 'user_annotation', 'python_function']
        found_items = self.parallel_search_values(data, key_path, target_values, num_threads=self.num_threads)
        return found_items


    def get_cuda_runtime_items(self, data):
        key_path = ['cat']
        target_values = ['cuda_runtime']
        found_items = self.parallel_search_values(data, key_path, target_values, num_threads=self.num_threads)
        return found_items


    def get_kernel_items(self, data):
        key_path = ['cat']
        target_values = ['kernel', 'gpu_memcpy', 'gpu_user_annotation']
        found_items = self.parallel_search_values(data, key_path, target_values, num_threads=self.num_threads)
        return found_items


    def get_items_by_name(self, item_names):
        key_path = ['name']
        target_values = item_names
        found_items = self.parallel_search_values(self.data, key_path, target_values, num_threads=self.num_threads)
        return found_items


    def get_sub_items_by_name(self, item_names, higher_item):
        sub_data = self.get_sub_data_by_item(higher_item)
        key_path = ['name']
        target_values = item_names
        found_items = self.parallel_search_values(sub_data, key_path, target_values, num_threads=self.num_threads)
        return found_items


    def get_sub_data_by_item(self, item):
        start_time = item.get('ts')
        end_time = start_time + item.get('dur')
        key_path = ['ts']
        sub_data = self.parallel_search_nested_range(self.data, key_path, start_time, end_time, num_threads=self.num_threads)
        return sorted(sub_data, key=lambda x: x.get('ts'))


    def get_cuda_runtimes_by_item(self, item):
        start_time = item.get('ts')
        end_time = start_time + item.get('dur')
        key_path = ['ts']
        cuda_runtimes = self.parallel_search_nested_range(self.cuda_runtime_items, key_path, start_time, end_time, num_threads=self.num_threads)
        return sorted(cuda_runtimes, key=lambda x: x.get('ts'))


    def get_cpu_ops_by_item(self, item):
        start_time = item.get('ts')
        end_time = start_time + item.get('dur')
        key_path = ['ts']
        cpu_ops = self.parallel_search_nested_range(self.cpu_op_items, key_path, start_time, end_time, num_threads=self.num_threads)
        return cpu_ops


    def get_kernels_by_cuda_runtimes(self, cuda_runtimes):
        key_path = ['args', 'External id']
        target_values = [d.get('args').get('External id') for d in cuda_runtimes]
        kernels = self.parallel_search_values(self.kernel_items, key_path, target_values, num_threads=self.num_threads)
        if len(kernels) > 0:
            kernels = sorted(kernels, key=lambda x: x.get('ts'))
            min_value = kernels[0].get('ts') - 1
            max_value = kernels[-1].get('ts') + 1 
            kernels = self.parallel_search_nested_range(self.kernel_items, ['ts'], min_value, max_value, num_threads=self.num_threads)
            return sorted(kernels, key=lambda x: x.get('ts'))
        return []


    def get_the_kernel_of_the_cuda_runtime(self, cuda_runtime, kernels):
        key_path = ['args', 'External id']
        target_values = [cuda_runtime.get('args').get('External id')]
        kernel_list = self.parallel_search_values(kernels, key_path, target_values, num_threads=self.num_threads)
        return kernel_list[0] if len(kernel_list) > 0 else None


    def get_the_cuda_runtime_by_kernel(self, kernel, cuda_runtimes):
        key_path = ['args', 'External id']
        target_values = [kernel.get('args').get('External id')]
        cuda_runtime_list = self.parallel_search_values(cuda_runtimes, key_path, target_values, num_threads=self.num_threads)
        return cuda_runtime_list[0] if len(cuda_runtime_list) > 0 else None


    def get_the_cpu_ops_of_the_cuda_runtime(self, cuda_runtime, cpu_ops):
        if cuda_runtime is None:
            return [None, None]
        cuda_runtime_start = cuda_runtime.get('ts')
        cuda_runtime_end = cuda_runtime_start + cuda_runtime.get('dur')
        left_cpu_ops = self.parallel_search_nested_range(cpu_ops, ['ts'], 0, cuda_runtime_start, num_threads=self.num_threads)
        if len(left_cpu_ops) > 0:
            in_range_cpu_ops = self.parallel_search_in_range_cpu_ops(left_cpu_ops, cuda_runtime_end, num_threads=self.num_threads)
        else:
            in_range_cpu_ops = [None]
        # highest cpu_op with biggest dur, lowest cpu_op with smallest dur 
        return [in_range_cpu_ops[-1], in_range_cpu_ops[0]] if len(in_range_cpu_ops) > 0 else [None, None]



    def get_gpu_runtime(self, item_names):
        results = []
        for item_name in item_names:
            target_items = self.get_items_by_name([item_name])
            print(f'total items: {len(target_items)}')
            temp_list = []
            if len(target_items) > 0:
                for target_item in target_items:
                    cuda_runtimes = self.get_cuda_runtimes_by_item(target_item)
                    kernels = self.get_kernels_by_cuda_runtimes(cuda_runtimes)
                    if len(kernels) > 0:
                        GPU_time = kernels[-1].get('ts') + kernels[-1].get('dur') - kernels[0].get('ts')
                        temp_list.append(GPU_time)
                if len(temp_list) > 0:
                    average = sum(temp_list) / len(temp_list)
                    minmum = min(temp_list)
                    # results.append(sum(temp_list) / len(temp_list))
                    results.append([average, minmum])
        return results


    def get_gpu_runtime_without_communication(self, item_names):        
        results = []
        for item_name in item_names:
            target_items = self.get_items_by_name([item_name])
            print(f'total items: {len(target_items)}')
            temp_list = []
            if len(target_items) > 0:
                for target_item in target_items:
                    cuda_runtimes = self.get_cuda_runtimes_by_item(target_item)
                    kernels = self.get_kernels_by_cuda_runtimes(cuda_runtimes)
                    if len(kernels) > 0:
                        GPU_time = kernels[-1].get('ts') + kernels[-1].get('dur') - kernels[0].get('ts')
                        
                        communication_time = 0
                        for kernel in kernels:
                            if 'ncclDev' in kernel.get('name'):
                                communication_time += kernel.get('dur')

                        temp_list.append(GPU_time - communication_time)
                if len(temp_list) > 0:
                    average = sum(temp_list) / len(temp_list)
                    minmum = min(temp_list)
                    # results.append(sum(temp_list) / len(temp_list))
                    results.append([average, minmum])
        return results


    def get_item_runtime(self, item_names):
        results = []
        for item_name in item_names:
            target_items = self.get_items_by_name([item_name])
            print(f'total items: {len(target_items)}')
            temp_list = []
            if len(target_items) > 0:
                temp_list = [item.get('dur') for item in target_items]
                if len(temp_list) > 0:
                    average = sum(temp_list) / len(temp_list)
                    minmum = min(temp_list)
                    results.append([average, minmum])
        return results




    def get_statistic_by_item_name(self, item_names):
        target_items = self.get_items_by_name(item_names)
        print(f'total items: {len(target_items)}')

        results_list = []
        max_len = 0

        count = 0
        for target_item in target_items:
            cuda_runtimes = self.get_cuda_runtimes_by_item(target_item)
            kernels = self.get_kernels_by_cuda_runtimes(cuda_runtimes)
            cpu_ops = self.get_cpu_ops_by_item(target_item)
            
            # list of [highest cpu_op, lowest cpu_op, kernel]
            results = []
            for kernel in kernels:
                cuda_runtime = self.get_the_cuda_runtime_by_kernel(kernel, cuda_runtimes)
                highest_cpu_op, lowest_cpu_op = self.get_the_cpu_ops_of_the_cuda_runtime(cuda_runtime, cpu_ops)
                results.append([highest_cpu_op, lowest_cpu_op, kernel])
            
            if len(results) > max_len:
                max_len = len(results)
            results_list.append(results)
            count += 1
            print(f'{count}/{len(target_items)} done !')

        longgest_results_list = []
        avaiable_count = 0
        for results in results_list:
            if len(results) == max_len:
                avaiable_count += 1
                longgest_results_list.append(results)

        print(f'avaiable results: {avaiable_count}')

        merged_results = []

        for i in range(max_len):
            merged_item = {'highest_cpu_op_list': [], 'lowest_cpu_op_list': [], 'kernel_list': []}
            for results in longgest_results_list:
                merged_item.get('highest_cpu_op_list').append(results[i][0])
                merged_item.get('lowest_cpu_op_list').append(results[i][1])
                merged_item.get('kernel_list').append(results[i][2])
            merged_results.append(merged_item)

        return merged_results


    # write csv row
    def write_result_to_csv(self, path, columns_name, result):
        if os.path.exists(path):
            with open(path, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(result)
        else:
            with open(path, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(columns_name)   
                writer.writerow(result)  


    def merged_results_to_CSV(self, save_path, merged_kernel_list):
        columns_name = ['cpu_op_0', 'cpu_op_0_id', 'cpu_op_0_input_dim', 'cpu_op_1', 'cpu_op_1_id', 'cpu_op_1_input_dim', 'kernel', 'kernel_id', 'kernel_overhead(us)', 'kernel_dur(us)']
        
        result = []
        merged_kernel_list_length = len(merged_kernel_list)
        for i in range(merged_kernel_list_length):
            kernel = merged_kernel_list[i]

            highest_cpu_op_name = kernel.get('highest_cpu_op_list')[0].get('name')  if kernel.get('highest_cpu_op_list')[0] is not None else 'None'
            highest_cpu_op_id = kernel.get('highest_cpu_op_list')[0].get('args').get('External id') if kernel.get('highest_cpu_op_list')[0] is not None else 'None'
            highest_cpu_op_input_dim = kernel.get('highest_cpu_op_list')[0].get('args').get('Input Dims') if kernel.get('highest_cpu_op_list')[0] is not None else 'None'
            
            lowest_cpu_op_name = kernel.get('lowest_cpu_op_list')[0].get('name') if kernel.get('lowest_cpu_op_list')[0] is not None else 'None'
            lowest_cpu_op_id = kernel.get('lowest_cpu_op_list')[0].get('args').get('External id') if kernel.get('lowest_cpu_op_list')[0] is not None else 'None'
            lowest_cpu_op_input_dim = kernel.get('lowest_cpu_op_list')[0].get('args').get('Input Dims') if kernel.get('lowest_cpu_op_list')[0] is not None else 'None'
        
            kernel_name = kernel.get('kernel_list')[0].get('name')
            kernel_id = kernel.get('kernel_list')[0].get('args').get('External id')
            
            kernel_dur_count = 0
            for gpu_kernel in kernel.get('kernel_list'):
                kernel_dur_count += gpu_kernel.get('dur')
            kernel_dur_average = kernel_dur_count / len(kernel.get('kernel_list'))

            if i == 0:
                kernel_overhead_average = 0
            else:
                last_kernel_list = merged_kernel_list[i - 1]
                kernel_overhead_count = 0
                for j in range(len(last_kernel_list.get('kernel_list'))):
                    kernel_overhead_count += kernel.get('kernel_list')[j].get('ts') - (last_kernel_list.get('kernel_list')[j].get('ts') + last_kernel_list.get('kernel_list')[j].get('dur'))
                kernel_overhead_average = kernel_overhead_count / len(last_kernel_list)

            result = [highest_cpu_op_name, highest_cpu_op_id, highest_cpu_op_input_dim, lowest_cpu_op_name, lowest_cpu_op_id, lowest_cpu_op_input_dim, kernel_name, kernel_id, kernel_overhead_average, kernel_dur_average]

            result = np.asarray(result, dtype="object")
            
            self.write_result_to_csv(save_path, columns_name, result)   



# def test():
#     data_path = './profilers/GPT_20B_2_N1_GPN1/2239682_11608/PP1_MP1_DP1_ZERO0/0/aisct02_1549904.1730684232021.pt.trace.json'
#     timelineDecoder = TimelineDecoder(data_path, num_threads=16)
#     item_names = ['deepspeed/runtime/activation_checkpointing/checkpointing.py(496): forward', 'CheckpointFunctionBackward']  
#     results = timelineDecoder.get_gpu_runtime(item_names)
#     print(results)

# def test2():
#     data_path = './profilers/GPT_20B_2_N1_GPN1/2239682_11608/PP1_MP1_DP1_ZERO0/0/aisct02_1549904.1730684232021.pt.trace.json'
#     timelineDecoder = TimelineDecoder(data_path, num_threads=16)
#     name = ['CheckpointFunctionBackward']  
#     merged_results = timelineDecoder.get_statistic_by_item_name(name)
#     save_path = './test_GPT_20B_2_N1_GPN1_bwd.csv'
#     timelineDecoder.merged_results_to_CSV(save_path, merged_results)




def get_decoders(path_list, unm_threads):
    decoder_list = []
    for path in path_list:
        print(f'loading: {path}')
        decoder_list.append(TimelineDecoder(path, num_threads=unm_threads))
    return decoder_list


# def gpu_runtimes(decoder_list, module_list, target_list, columns_name, save_path):

#     results_dict = {}
#     for target in target_list:
#         results = []
#         for decoder in decoder_list:
#             results.append(decoder.get_gpu_runtime([target]))
#         results_dict[target] = results

#     # write results to csv
#     for i in range(len(module_list)):
#         result = [module_list[i], target_list[i], results_dict.get(target_list[i])]
#         tools.write_one_result_to_csv(save_path, columns_name, result)

#     # return results_dict


def gpu_runtimes(decoder_list, module_list, target_list, columns_name, save_path):

    for i in range(len(module_list)):
        module_name = module_list[i]
        target = target_list[i]

        average_results = []
        min_results = []

        for decoder in decoder_list:
            print(f'{module_name}, GPU:{len(average_results)}')
            temp_results = decoder.get_gpu_runtime([target])
            if len(temp_results) > 0:
                result = temp_results[0]
                # results.append(result[0])
                average_results.append(result[0])
                min_results.append(result[1])
            else:
                average_results.append(0)
                min_results.append(0)

        average_list_for_write = [module_name, target, average_results]
        min_list_for_write = [module_name, target, min_results]

        tools.write_one_result_to_csv(save_path[0], columns_name, average_list_for_write)
        tools.write_one_result_to_csv(save_path[1], columns_name, min_list_for_write)


def item_runtimes(decoder_list, module_list, target_list, columns_name, save_path):

    for i in range(len(module_list)):
        module_name = module_list[i]
        target = target_list[i]

        average_results = []
        min_results = []

        for decoder in decoder_list:
            print(f'{module_name}, GPU:{len(average_results)}')
            temp_results = decoder.get_item_runtime([target])
            if len(temp_results) > 0:
                result = temp_results[0]
                # results.append(result[0])
                average_results.append(result[0])
                min_results.append(result[1])
            else:
                average_results.append(0)
                min_results.append(0)

        average_list_for_write = [module_name, target, average_results]
        min_list_for_write = [module_name, target, min_results]

        tools.write_one_result_to_csv(save_path[0], columns_name, average_list_for_write)
        tools.write_one_result_to_csv(save_path[1], columns_name, min_list_for_write)


def operator_statistics(decoder_list, module_list, target_list, save_folder):

    encoder_index = 0
    for decoder in decoder_list:
        folder = save_folder + '/' + str(encoder_index) 
        os.makedirs(folder, exist_ok=True)
        for i in range(len(module_list)):
            module_name = module_list[i]
            target = target_list[i]
            merged_results = decoder.get_statistic_by_item_name([target])
            save_path = folder +  '/' + module_name + '.csv'
            decoder.merged_results_to_CSV(save_path, merged_results)
        encoder_index += 1


def modules_gpu_runtime(decoder_list, save_folder):

    '''
    (o): the module can use item dur
    (x): the module cannot use item dur  
    vista:
    (x) pp_p2p/send_activation: deepspeed/runtime/pipe/engine.py(948): _exec_send_activations   
    (x) mp_allreduce: _ReduceFromModelParallelRegion
    (x) dp_allreduce/reduce_grads: deepspeed/runtime/pipe/engine.py(270): _exec_reduce_grads
    (x) dp_allgather: deepspeed/runtime/utils.py(940): all_gather_all_partitions
    (o) encoder_fwd: deepspeed/runtime/activation_checkpointing/checkpointing.py(496): forward
    (o) encoder_bwd: CheckpointFunctionBackward
    (x) stage_fwd: deepspeed/runtime/pipe/module.py(319): forward
    (x) stage_bwd: deepspeed/runtime/pipe/engine.py(710): _exec_backward_pass
    (x) optimizer: deepspeed/runtime/pipe/engine.py(1140): _exec_optimizer_step    (dp_allgather is included in this module)
    (o) train_batch: deepspeed/runtime/pipe/engine.py(312): train_batch
    '''

    # module_list = ['pp_p2p', 'mp_allreduce', 'dp_allreduce', 'dp_allgather', 'encoder_fwd', 'encoder_bwd', 'stage_fwd', 'stage_bwd', 'optimizer', 'train_batch']

    # target_list = ['deepspeed/runtime/pipe/engine.py(948): _exec_send_activations',    
    #             '_ReduceFromModelParallelRegion',
    #             'deepspeed/runtime/pipe/engine.py(270): _exec_reduce_grads',
    #             'deepspeed/runtime/utils.py(940): all_gather_all_partitions',
    #             'deepspeed/runtime/activation_checkpointing/checkpointing.py(496): forward',
    #             'CheckpointFunctionBackward',
    #             'deepspeed/runtime/pipe/module.py(319): forward',
    #             'deepspeed/runtime/pipe/engine.py(710): _exec_backward_pass',
    #             'deepspeed/runtime/pipe/engine.py(1140): _exec_optimizer_step',
    #             'deepspeed/runtime/pipe/engine.py(312): train_batch'
    #             ]

    columns_name = ['module', 'function', 'gpu_runtime(us)']
    save_path = [save_folder + '/module_times_average.csv', save_folder + '/module_times_min.csv']

    gpu_module_list = ['pp_p2p', 'mp_allreduce', 'dp_allreduce', 'dp_allgather', 'stage_fwd', 'stage_bwd', 'optimizer']

    gpu_target_list = ['deepspeed/runtime/pipe/engine.py(948): _exec_send_activations',    
                '_ReduceFromModelParallelRegion',
                'deepspeed/runtime/pipe/engine.py(270): _exec_reduce_grads',
                'deepspeed/runtime/utils.py(940): all_gather_all_partitions',
                'deepspeed/runtime/pipe/module.py(319): forward',
                'deepspeed/runtime/pipe/engine.py(710): _exec_backward_pass',
                'deepspeed/runtime/pipe/engine.py(1140): _exec_optimizer_step',
                ]

    item_module_list = ['encoder_fwd', 'encoder_bwd', 'update']

    item_target_list = ['deepspeed/runtime/activation_checkpointing/checkpointing.py(496): forward',
                'CheckpointFunctionBackward',
                'deepspeed/runtime/pipe/engine.py(312): train_batch'
                ]

    item_runtimes(decoder_list, item_module_list, item_target_list, columns_name, save_path)
    gpu_runtimes(decoder_list, gpu_module_list, gpu_target_list, columns_name, save_path)


def get_optimizer(decoder_list, save_folder):
    
    columns_name = ['module', 'function', 'average_runtime(us)', 'min_runtime(us)']

    module_name = 'optimizer'
    target = 'deepspeed/runtime/pipe/engine.py(1140): _exec_optimizer_step'

    average_results = []
    min_results = []

    for decoder in decoder_list:
        print(f'{module_name}, GPU:{len(average_results)}')
        temp_results = decoder.get_gpu_runtime_without_communication([target])
        if len(temp_results) > 0:
            result = temp_results[0]
            # results.append(result[0])
            average_results.append(result[0])
            min_results.append(result[1])
        else:
            average_results.append(0)
            min_results.append(0)

    list_for_write = [module_name, target, average_results, min_results]
    
    save_path = save_folder +  '/' + '(' + tools.get_current_time() + ')_optimizer.csv'

    tools.write_one_result_to_csv(save_path, columns_name, list_for_write)




def modules_operators_runtime(decoder_list, save_folder):
    # module_list = ['encoder_fwd', 'encoder_bwd', 'reduce_grads', 'optimizer_step', 'fwd', 'bwd']

    # #for VISTA
    # target_list = ['deepspeed/runtime/activation_checkpointing/checkpointing.py(496): forward', 
    #                'CheckpointFunctionBackward',
    #                'deepspeed/runtime/pipe/engine.py(270): _exec_reduce_grads', 
    #                'deepspeed/runtime/pipe/engine.py(1140): _exec_optimizer_step', 
    #                'deepspeed/runtime/pipe/module.py(319): forward', 
    #                'deepspeed/runtime/pipe/engine.py(710): _exec_backward_pass'
    #                ]

    module_list = ['pp_p2p', 'dp_allreduce', 'optimizer']

    #for VISTA
    target_list = ['deepspeed/runtime/pipe/engine.py(948): _exec_send_activations', 
                    'deepspeed/runtime/pipe/engine.py(270): _exec_reduce_grads',
                    'deepspeed/runtime/pipe/engine.py(1140): _exec_optimizer_step']

    operator_statistics(decoder_list, module_list, target_list, save_folder + '/best_batch_statistics' )


def decoding(path_list, save_folder, unm_threads):
    decoder_list = get_decoders(path_list, unm_threads)
    modules_gpu_runtime(decoder_list, save_folder)
    modules_operators_runtime(decoder_list, os.path.dirname(save_folder))
    # get_optimizer(decoder_list, save_folder)



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


def run_decoder(target_path, unm_threads):
    path_list = get_stages_profiler_path(target_path)
    # save_folder = os.path.dirname(target_path)
    save_folder = target_path
    # unm_threads = 16
    decoding(path_list, save_folder, unm_threads)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", default=16, type=int,
                        help="Number of threads for decoding")
    parser.add_argument("--target_path", default='', type=str,
                        help="Path of profilers' folder")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments() 
    run_decoder(args.target_path, args.threads)

    # python vista_parallel_timeline_decoder_new.py --threads 64 --target_path ./vista/GPT_350M_8_1/91457/PP2_MP2_DP2_ZERO1
    # python vista_parallel_timeline_decoder_new.py --threads 64 --target_path ./vista/GPT_20B_128_1/127321/PP4_MP4_DP8_ZERO1
    # python vista_parallel_timeline_decoder_new.py --threads 64 --target_path ./vista/GPT_20B_128_1/127322/PP4_MP8_DP4_ZERO1
    # python vista_parallel_timeline_decoder_new.py --threads 64 --target_path ./vista/GPT_20B_128_1/127323/PP8_MP4_DP4_ZERO1
    # python vista_parallel_timeline_decoder_new.py --threads 64 --target_path ./perlmutter/GPT_20B_8_4_4_32_4/31697099/PP8_MP4_DP4_ZERO1
    # python vista_parallel_timeline_decoder_new.py --threads 64 --target_path ./vista/llama_13B_64_1/136645/PP4_MP8_DP2_ZERO1

    # python vista_parallel_timeline_decoder_new.py --threads 32 --target_path /mnt/vstor/CSE_CSDS_VXC204/bxz297/ICICLE/AI4CI/AI4CI3/GPUModeling/3D_parallelism_prediction/vista/llama_13B_64_1/136645/PP4_MP8_DP2_ZERO1


    # decoder the best_batch
    # python vista_parallel_timeline_decoder_new.py --threads 16 --target_path ./vista/GPT_350M_8_1/91457/best_batch
    # python vista_parallel_timeline_decoder_new.py --threads 16 --target_path ./vista/GPT_20B_128_1/127322/best_batch
    # python vista_parallel_timeline_decoder_new.py --threads 16 --target_path ./vista/GPT_20B_128_1/100697/best_batch


