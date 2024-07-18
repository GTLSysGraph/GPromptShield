from prompt_graph.data import load4node_attack_shot_index


dataset_name   = 'Cora'
attack_method  = 'Meta_Self-0.2'
shot_num       =  5
run_split      =  1


data, dataset = load4node_attack_shot_index(dataset_name, attack_method, shot_num = shot_num, run_split= run_split)