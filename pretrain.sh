python MyPretrain.py \
--task 'GraphCL' \
--dataset_name 'Cora_ml' \
--preprocess_method 'none' \
--gnn_type 'GCN' \
--hid_dim 128 \
--num_layer 2 \
--epochs 200 \
--seed 787 \
--device 1

# 'GraphCL'  'Edgepred_Gprompt' 'NodeMultiGprompt' 'GraphMAE'

#########################################
# GraphCL hid_dim
# Cora 256 
# Citeseer 64 比较好 用32 256不起作用 GraphCL训小图不要用那么多划分num_parts
# coraml 64

# 'Edgepred_Gprompt' hid_dim
# Cora 256 
# Citeseer 64 比较好


# NodeMultiGprompt hid_dim
# Cora 256 
# Citeseer 128 比较好
#########################################



#########################################
# GraphMAE
# Citeseer 256很不错
# CoraML  64