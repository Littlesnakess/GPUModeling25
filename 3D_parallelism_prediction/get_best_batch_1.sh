#!/bin/bash




# python get_the_best_batch.py --threads 16 --target_path ./vista/GPT_20B_128_1/127322/PP4_MP8_DP4_ZERO1




python get_the_best_batch.py --threads 16 --target_path ./vista/GPT_20B_128_1/100696/PP4_MP4_DP8_ZERO1
python get_the_best_batch.py --threads 16 --target_path ./vista/GPT_20B_128_1/100697/PP4_MP8_DP4_ZERO1
python get_the_best_batch.py --threads 16 --target_path ./vista/GPT_20B_128_1/100698/PP8_MP4_DP4_ZERO1

python get_the_best_batch.py --threads 16 --target_path ./vista/GPT_20B_128_1/127321/PP4_MP4_DP8_ZERO1
python get_the_best_batch.py --threads 16 --target_path ./vista/GPT_20B_128_1/127322/PP4_MP8_DP4_ZERO1
python get_the_best_batch.py --threads 16 --target_path ./vista/GPT_20B_128_1/127323/PP8_MP4_DP4_ZERO1

python get_the_best_batch.py --threads 16 --target_path ./vista/llama_13B_64_1/94078/PP4_MP8_DP2_ZERO1
python get_the_best_batch.py --threads 16 --target_path ./vista/llama_13B_64_1/136645/PP4_MP8_DP2_ZERO1
python get_the_best_batch.py --threads 16 --target_path ./vista/llemma_7B_16_1/91703/PP4_MP2_DP2_ZERO1


python get_the_best_batch.py --threads 16 --target_path ./perlmutter/GPT_20B_4_4_8_32_4/31694544/PP4_MP4_DP8_ZERO1
python get_the_best_batch.py --threads 16 --target_path ./perlmutter/GPT_20B_4_8_4_32_4/31697107/PP4_MP8_DP4_ZERO1
python get_the_best_batch.py --threads 16 --target_path ./perlmutter/GPT_20B_8_4_4_32_4/31697099/PP8_MP4_DP4_ZERO1
python get_the_best_batch.py --threads 16 --target_path ./perlmutter/llama_13B_4_8_2_16_4/31674706/PP4_MP8_DP2_ZERO1
python get_the_best_batch.py --threads 16 --target_path ./perlmutter/llemma_7B_4_2_2_4_4/31664313/PP4_MP2_DP2_ZERO1