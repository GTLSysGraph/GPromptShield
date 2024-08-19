python MyPretrain.py \
--task NodeMultiGprompt \
--dataset_name 'Cora' \
--preprocess_method 'none' \
--gnn_type 'GCN' \
--hid_dim 256 \
--num_layer 2 \
--epochs 200 \
--seed 42 \
--device 1