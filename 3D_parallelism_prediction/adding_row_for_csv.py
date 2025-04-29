import pandas as pd

# Load the CSV file
file_path = "/mnt/vstor/CSE_CSDS_VXC204/bxz297/ICICLE/AI4CI/AI4CI3/GPUModeling/3D_parallelism_prediction/Data/operators_GH200/sampling_data/NVIDIAGH200120GB_ScaledUpperTriangMaskedSoftmax_fp16.csv"

save_path = "/mnt/vstor/CSE_CSDS_VXC204/bxz297/ICICLE/AI4CI/AI4CI3/GPUModeling/3D_parallelism_prediction/Data/operators_GH200/NVIDIAGH200120GB_ScaledUpperTriangMaskedSoftmax_fp16.csv"

df = pd.read_csv(file_path)

# Insert a new column at the first position
new_column_name = "mp"
new_column_values = [1] * len(df)  # Replace with your values

df.insert(0, new_column_name, new_column_values)  # Insert at the first position

# Save the updated CSV
df.to_csv(save_path, index=False)