python MyPretrain.py \
--task 'GraphCL' \
--dataset_name 'PubMed' \
--preprocess_method 'none' \
--gnn_type 'GCN' \
--hid_dim 256 \
--num_layer 2 \
--epochs 200 \
--seed 1 \
--device 1

# 'GraphCL'  'Edgepred_Gprompt' 'NodeMultiGprompt' 'GraphMAE'

#########################################
# GraphCL hid_dim
# Cora 256 
# Citeseer 64          GraphCL num_parts=5
# coraml 64

# 'Edgepred_Gprompt' hid_dim
# Cora 256 
# Citeseer 64 


# NodeMultiGprompt hid_dim
# Cora 256 
# Citeseer 128 
#########################################



#########################################
# GraphMAE
# Citeseer 256
# CoraML  64