# GPUModeling

This repository is a tool for predicting the time cost of distributed training of LLMs.



# Environment and Dependencies

This is not an application that can be used freely yet. Please do not modify the existing directory structure of the program.

To install the ing basic dependencies, run:

```bash
cd GPUModeling25  # repository root
pip install -r requirements.txt
```

from the repository root.


# Profiling Data

## Preparation

Profiling files can be download at https://doi.org/10.5281/zenodo.15288792 and save them under folder 3D_parallelism_prediction. Unzip all the zip files and keep the directory structure. Fully decompressing perlmutter.zip needs more than 36.69GB and vista.zip needs more than 178.84GB. 


## Overall Decoding
The functional scrpit is **`vista_parallel_timeline_decoder_new.py`**. 

```bash
cd GPUModeling25/3D_parallelism_prediction  # Predictor folder

# for one experiments e.g. GPT_20B_4_4_8 on Perlmutter
python vista_parallel_timeline_decoder_new.py \
--target_path ./perlmutter/GPT_20B_4_4_8_32_4/31694544/PP4_MP4_DP8_ZERO1

# for all experiments run the bash script (will cost about one week)
./run_decoder.sh
```

## Best Loop Decoding
The best loop refers to the fastest iteration for a single parameter update. The functional script is **`get_the_best_batch.py`**. The **`best_batch`** folder contains the extracted profiling files, which are compressed for uploading to GitHub. You need to unzip them before running the decoding script.
```bash
cd GPUModeling25/3D_parallelism_prediction  # Predictor folder

# get the json items of best loop for all experiments (will cost about one day)
./get_best_batch_1.sh

# decode best loop for all experiments (will cost about one day)
./decode_best_batch_1.sh
```

# Oprator Sampling
The **`Kernel_sampling`** folder contains the code for sampling computational operations, while the **`NCCL_sampling`** folder is used for sampling communication operations.

## Computational Kernels
```bash
cd GPUModeling25/Kernel_sampling  # Predictor folder

# Sampling for all kernels (will cost about two days)
./run_collection.sh
```

## Communicational Kernels
```bash
# Example Sampling script for p2p communicational kernels which needs to be integrated with slurm script

# Get master address and port
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Get the number of nodes
NNODES=$(scontrol show hostname $SLURM_NODELIST | wc -l)

srun --export=ALL,CUDA_VISIBLE_DEVICES=0 torchrun \
--nnodes $NNODES \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
sampling_controller.py \
--config_path ./configs/test/p2p.yml \   # indicate operations
--precision fp16 \
--parts 1 \
--part 1
```

# Prediction

## Operation Level Prediction
```bash
cd GPUModeling25/3D_parallelism_prediction  # Predictor folder

# Generate operation level predictions in the folder of profiling files
python 3D_prediction.py
```

## Component Level and Overall Errors Caculations
```bash
cd GPUModeling25/3D_parallelism_prediction  # Predictor folder

# Generate operation level predictions in the folder of profiling files
python results_compare.py
```

## Generate the Error Table (Table 9 in paper) 
```bash
cd GPUModeling25/3D_parallelism_prediction  # Predictor folder

# Generate operation level predictions in the folder of profiling files
python get_the_component_statistic.py
```
This will create a .csv file in folder **`statistics`**.

## Generate the Component Proportions Figure (Figure 4 in paper) 
```bash
cd GPUModeling25/3D_parallelism_prediction  # Predictor folder

# Generate operation level predictions in the folder of profiling files
python component_portion_percentage.py
```
This will create a .png figure in folder **`statistics`**.


# Others
## Hyper-Parameter Tuning for Regressors
```bash
cd GPUModeling25/3D_parallelism_prediction  # Predictor folder

# For computation kernels
python operator_tunning.py

# For communication kernels
python nccl_tunning.py --target_path ./Data/nccl_perlmutter/required_renamed
```
This will create a .csv file under both dataset's folder that includes the tunning results for each of them.

## Table for HPCs' Performance Stability (Table 8 in pape)
```bash
cd GPUModeling25/3D_parallelism_prediction  # Predictor folder

# Generate operation level predictions in the folder of profiling files
python get_the_iteration_statistic.py
```
This will create a .png figure in folder **`statistics`**.
