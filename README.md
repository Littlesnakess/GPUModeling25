# GPUModeling

This repository is a tool for predicting the time cost of distributed training of LLMs.



# Environment and Dependencies

This is not an application that can be used freely yet. Please do not modify the existing directory structure of the program.

To install the ing basic dependencies, run:

```bash
cd GPUModeling25  # repository root
pip install -r requirements/requirements.txt
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
Best loop means the fastest iteration of onece parameter update. The functional scrpit is **`get_the_best_batch.py`**. 

```bash
cd GPUModeling25/3D_parallelism_prediction  # Predictor folder

# get the json items of best loop for all experiments (will cost about one day)
./get_best_batch_1.sh

# decode best loop for all experiments (will cost about one day)
./decode_best_batch_1.sh
```


