from prompt_graph.pretrain import GraphCL,Edgepred_Gprompt, GraphPrePrompt, NodePrePrompt, GraphMAE
from prompt_graph.utils import seed_everything
from prompt_graph.utils import mkdir, get_args

args = get_args()
for seed in args.seed:
    seed_everything(seed)
    mkdir('./pre_trained_model/')

    if args.task == 'GraphCL':
        pt = GraphCL(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, preprocess_method = args.preprocess_method, device=args.device)
        # Cora 256
    if args.task == 'Edgepred_Gprompt':
        pt = Edgepred_Gprompt(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device)

    if args.task == 'NodeMultiGprompt':
        # epoch 在内部自己设置
        nonlinearity = 'prelu'
        pt = NodePrePrompt(dataset_name=args.dataset_name, n_h=args.hid_dim, activation=nonlinearity, a1 = 0.9, a2= 0.9, a3= 0.1, a4= 0.0001, num_layers_num = args.num_layer,  p = 0.3, device=args.device)

    if args.task == 'GraphMAE':
        pt = GraphMAE(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device,
                    mask_rate=0.75, drop_edge_rate=0.0, replace_rate=0.1, loss_fn='sce', alpha_l=2)

    # if args.task == 'GraphMultiGprompt':
    #     nonlinearity = 'prelu'
    #     pt = GraphPrePrompt(dataset_name=args.dataset_name, n_h=args.hid_dim, activation=nonlinearity, a1 = 0.9, a2= 0.9, a3= 0.1, num_layers_num = 1, p = 0.3, device=args.device)

    pt.pretrain() # batch_size=100