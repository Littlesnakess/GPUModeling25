

# GPT_20B_PP4_MP4_DP8_ZERO1

target="./perlmutter/GPT_20B_4_4_8_32_4/31694544/PP4_MP4_DP8_ZERO1"
python_app="./vista_parallel_timeline_decoder_new.py"
python $python_app --target_path $target


target="./vista/GPT_20B_128_1/100696/PP4_MP4_DP8_ZERO1"
python_app="./vista_parallel_timeline_decoder_new.py"
python $python_app --target_path $target



# GPT_20B_PP4_MP8_DP4_ZERO1

target="./perlmutter/GPT_20B_4_8_4_32_4/31697107/PP4_MP8_DP4_ZERO1"
python_app="./vista_parallel_timeline_decoder_new.py"
python $python_app --target_path $target


target="./vista/GPT_20B_128_1/100697/PP4_MP8_DP4_ZERO1"
python_app="./vista_parallel_timeline_decoder_new.py"
python $python_app --target_path $target



# GPT_20B_PP8_MP4_DP4_ZERO1

target="./perlmutter/GPT_20B_8_4_4_32_4/31697099/PP8_MP4_DP4_ZERO1"
python_app="./vista_parallel_timeline_decoder_new.py"
python $python_app --target_path $target


target="./vista/GPT_20B_128_1/100698/PP8_MP4_DP4_ZERO1"
python_app="./vista_parallel_timeline_decoder_new.py"
python $python_app --target_path $target



# llama_13B_4_8_2

target="./perlmutter/llama_13B_4_8_2_16_4/31674706/PP4_MP8_DP2_ZERO1"
python_app="./vista_parallel_timeline_decoder_new.py"
python $python_app --target_path $target


target="./vista/llama_13B_64_1/94078/PP4_MP8_DP2_ZERO1"
python_app="./vista_parallel_timeline_decoder_new.py"
python $python_app --target_path $target



# llemma_7B_4_2_2

target="./perlmutter/llemma_7B_4_2_2_4_4/31664313/PP4_MP2_DP2_ZERO1"
python_app="./vista_parallel_timeline_decoder_new.py"
python $python_app --target_path $target


target="./vista/llemma_7B_16_1/91703/PP4_MP2_DP2_ZERO1"
python_app="./vista_parallel_timeline_decoder_new.py"
python $python_app --target_path $target