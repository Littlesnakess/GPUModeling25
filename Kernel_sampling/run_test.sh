#!/bin/bash

root_path="/mnt/vstor/CSE_CSDS_VXC204/bxz297/ICICLE/AI4CI/AI4CI3/GPUModeling/Kernel_sampling"

python $root_path/sampling_controller.py --config_path $root_path/configs/test/baddbmm.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/bmm.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/crossentropy.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/embedding.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/fillmask.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/gelu.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/layernorm.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/linear_final.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/linear1.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/linear2.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/linear3.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/linear4.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/RoPE.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/softmax.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/flash_atten.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/RMSlayernorm.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/res_add.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/firstStage_optimizer.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/middleStage_optimizer.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/test/lastStage_optimizer.yml --precision fp16