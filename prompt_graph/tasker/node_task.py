import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data   import Data
from prompt_graph.utils import constraint,  center_embedding, Gprompt_tuning_loss, process, cmd
from prompt_graph.evaluation import GPPTEva, GNNNodeEva, GpromptEva, MultiGpromptEva, GPFEva, AllInOneEva, RobustPromptInductiveEva, RobustPromptTranductiveEva
from prompt_graph.data import induced_graphs, split_induced_graphs,load4node_shot_index, load4node_attack_shot_index, load4node_attack_specified_shot_index


from .task import BaseTask
import time
import warnings

import pickle
import os
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix
from tqdm import tqdm

warnings.filterwarnings("ignore")

class NodeTask(BaseTask):
      def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.task_type = 'NodeTask'

            if self.attack_downstream:
                  assert self.attack_method != 'None', 'No specific attacks were designated.'
                  if self.specified:
                        # 对指定的train/val/test划分方式进行攻击，因为一些方法对不同的划分会产生不同的分布
                        print("load LLC or attacked data with specified split")
                        self.load_specified_attack_data()
                  else:
                        # 对默认的train/val/test划分方式进行攻击
                        print('load LLC or attacked data with default split')
                        self.load_attack_data()
            else:
                  print('load raw data')
                  self.load_data()

            self.initialize_gnn()
            self.initialize_prompt()
            self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim), torch.nn.Softmax(dim=1)).to(self.device)
            self.initialize_optimizer()
      


      def process_multigprompt_data(self, data):
            adj = to_scipy_sparse_matrix(data.edge_index).tocsr()
            # Convert features to dense format and then to scipy sparse matrix in lil format
            features = sp.lil_matrix(data.x.numpy())
            # Convert labels to one-hot encoding
            labels = np.zeros((data.num_nodes, data.y.max().item() + 1))
            labels[np.arange(data.num_nodes), data.y.numpy()] = 1

            # adj, features, labels = process.load_data(self.dataset_name)
            # adj, features, labels = process.load_data(self.dataset_name)  
            self.input_dim = features.shape[1]
            self.output_dim = labels.shape[1]

            features, _ = process.preprocess_features(features)
            self.sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj).to(self.device)
            self.labels = torch.FloatTensor(labels[np.newaxis])
            self.features = torch.FloatTensor(features[np.newaxis]).to(self.device)
            # print("labels",labels)
            print("adj",self.sp_adj.shape)
            print("feature",features.shape)


      def load_specified_attack_data(self):
            # 加载攻击指定数据划分后的图数据
            self.data, self.dataset = load4node_attack_specified_shot_index(self.dataset_name, self.attack_method, shot_num = self.shot_num, run_split= self.run_split)
            if self.prompt_type == 'MultiGprompt':
                  self.process_multigprompt_data(self.data)
            else:
                  self.input_dim = self.data.x.shape[1]
                  self.output_dim = self.dataset.num_classes
            # print(self.data)
            # print(self.data.train_mask.nonzero().squeeze())
            # print("loaded specified attack data")
            # quit()
                  
            if self.prompt_type in ['All-in-one','Gprompt', 'GPF', 'GPF-plus','RobustPrompt_I']:
                  file_dir = './data_attack_fewshot/{}/shot_{}/{}/induced_graph/{}'.format(self.dataset_name, str(self.shot_num), str(self.run_split), self.attack_method)
                  file_path = os.path.join(file_dir, 'induced_graph.pkl')

                  # 注意，换shot num的时候要把induced graph删掉
                  if os.path.exists(file_path):
                        print('Begin load induced_graphs with specified shot {} and run split {} under {}.'.format(str(self.shot_num), str(self.run_split), self.attack_method))
                        with open(file_path, 'rb') as f:
                              graphs_dict = pickle.load(f)
                        self.train_dataset = graphs_dict['train_graphs']
                        self.test_dataset = graphs_dict['test_graphs']
                        self.val_dataset = graphs_dict['val_graphs']
                  else:
                        os.makedirs(file_dir, exist_ok=True) 
                        print('Begin split induced_graphs with specified shot {} and run split {} under {}.'.format(str(self.shot_num), str(self.run_split), self.attack_method))
                        split_induced_graphs(self.dataset_name, self.data, file_path, smallest_size=100, largest_size=300)
                        with open(file_path, 'rb') as f:
                              graphs_dict = pickle.load(f)
                        self.train_dataset = graphs_dict['train_graphs']
                        self.test_dataset = graphs_dict['test_graphs']
                        self.val_dataset = graphs_dict['val_graphs']
                  
                  # add by ssh 把除了训练集induced graph之外的图作为一个remaining dataset,为了讨论分布的一致问题
                  if self.prompt_type == 'RobustPrompt_I':
                        print("Combine the val and the test dataset to study the distribution shift problem! ")
                        self.remaining_dataset = self.val_dataset + self.test_dataset

            else:
                  self.data.to(self.device)

            # train_indices = self.data.train_mask.nonzero().squeeze()
            # print(train_indices)
            # print(self.data.y[train_indices])
            # quit()

      def load_attack_data(self):
            # 加载默认数据划分攻击后的图数据
            self.data, self.dataset = load4node_attack_shot_index(self.dataset_name, self.attack_method, shot_num = self.shot_num, run_split= self.run_split)
            
            if self.prompt_type == 'MultiGprompt':
                  self.process_multigprompt_data(self.data)
            else:
                  self.input_dim = self.data.x.shape[1]
                  self.output_dim = self.dataset.num_classes

            if self.prompt_type in ['All-in-one','Gprompt', 'GPF', 'GPF-plus','RobustPrompt_I']:
                  file_dir = './data_attack/{}/{}/induced_graph/shot_{}/{}'.format(self.dataset_name, self.attack_method, str(self.shot_num), str(self.run_split))
                  file_path = os.path.join(file_dir, 'induced_graph.pkl')

                  # 注意，换shot num的时候要把induced graph删掉
                  if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                              graphs_dict = pickle.load(f)
                        self.train_dataset = graphs_dict['train_graphs']
                        self.test_dataset = graphs_dict['test_graphs']
                        self.val_dataset = graphs_dict['val_graphs']
                  else:
                        os.makedirs(file_dir, exist_ok=True) 
                        print('Begin split_induced_graphs.')
                        split_induced_graphs(self.dataset_name, self.data, file_path, smallest_size=100, largest_size=300)
                        with open(file_path, 'rb') as f:
                              graphs_dict = pickle.load(f)
                        self.train_dataset = graphs_dict['train_graphs']
                        self.test_dataset = graphs_dict['test_graphs']
                        self.val_dataset = graphs_dict['val_graphs']
                  
                  # add by ssh 把除了训练集induced graph之外的图作为一个remaining dataset,为了讨论分布的一致问题
                  if self.prompt_type == 'RobustPrompt_I':
                        print("Combine the val and the test dataset to study the distribution shift problem! ")
                        self.remaining_dataset = self.val_dataset + self.test_dataset

            else:
                  self.data.to(self.device)



      def load_data(self):
            self.data, self.dataset = load4node_shot_index(self.dataset_name, preprocess_method = 
            self.preprocess_method, shot_num = self.shot_num, run_split= self.run_split)

            # train_indices = self.data.train_mask.nonzero().squeeze()
            # print(train_indices)
            # print(self.data.y[train_indices])
            # quit()

            if self.prompt_type == 'MultiGprompt':
                  self.process_multigprompt_data(self.data)
            else:
                  self.input_dim = self.data.x.shape[1]
                  self.output_dim = self.dataset.num_classes


            if self.prompt_type in ['All-in-one','Gprompt', 'GPF', 'GPF-plus']:
                  # file_dir = './data/{}/induced_graph/shot_{}/{}'.format(self.dataset_name, str(self.shot_num), str(self.run_split))
                  # file_path = os.path.join(file_dir, 'induced_graph.pkl')

                  file_dir = './data_fewshot/{}/shot_{}/{}/induced_graph/'.format(self.dataset_name, str(self.shot_num), str(self.run_split))
                  file_path = os.path.join(file_dir, 'induced_graph.pkl')


                  # 注意，换shot num的时候要把induced graph删掉
                  if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                              graphs_dict = pickle.load(f)
                        self.train_dataset = graphs_dict['train_graphs']
                        self.test_dataset = graphs_dict['test_graphs']
                        self.val_dataset = graphs_dict['val_graphs']
                  else:
                        os.makedirs(file_dir, exist_ok=True) 
                        print('Begin split_induced_graphs.')
                        split_induced_graphs(self.dataset_name, self.data, file_path, smallest_size=100, largest_size=300)
                        with open(file_path, 'rb') as f:
                              graphs_dict = pickle.load(f)
                        self.train_dataset = graphs_dict['train_graphs']
                        self.test_dataset = graphs_dict['test_graphs']
                        self.val_dataset = graphs_dict['val_graphs']
            else:
                  self.data.to(self.device)
            quit()

      def train(self, data):
            self.gnn.train()
            self.optimizer.zero_grad() 
            out = self.gnn(data.x, data.edge_index, batch=None) 
            out = self.answering(out)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()  
            self.optimizer.step()  
            return loss.item()
 

      def AllInOneTrain(self, train_loader):
            #we update answering and prompt alternately.
            
            answer_epoch = 20  # 50 80
            prompt_epoch = 20  # 50 80
            
            # tune task head
            self.answering.train()
            self.prompt.eval()
            for epoch in range(1, answer_epoch + 1):
                  answer_loss = self.prompt.Tune(train_loader, self.gnn,  self.answering, self.criterion, self.answer_opi, self.device)
                  print(("frozen gnn | frozen prompt | *tune answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, answer_loss)))

            # tune prompt
            self.answering.eval()
            self.prompt.train()
            for epoch in range(1, prompt_epoch + 1):
                  pg_loss = self.prompt.Tune( train_loader,  self.gnn, self.answering, self.criterion, self.pg_opi, self.device)
                  print(("frozen gnn | *tune prompt |frozen answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, pg_loss)))
            
            return pg_loss




      def MultiGpromptTrain(self, pretrain_embs, train_lbls, train_idx):
            self.DownPrompt.train()
            self.optimizer.zero_grad()
            prompt_feature = self.feature_prompt(self.features)
            # prompt_feature = self.feature_prompt(self.data.x)
            # embeds1 = self.gnn(prompt_feature, self.data.edge_index)
            embeds1= self.Preprompt.gcn(prompt_feature, self.sp_adj , True, False)
            pretrain_embs1 = embeds1[0, train_idx]
            # pretrain_embs  是使用preprompt的gcn生成的还没有加prompt的embs的train embs
            # pretrain_embs1 是加了提示后的train features生成的train embs
            logits = self.DownPrompt(pretrain_embs,pretrain_embs1, train_lbls, 1).float().to(self.device) # 1 shot  Cora torch.Size([7, 7])
            loss = self.criterion(logits, train_lbls)           
            loss.backward(retain_graph=True)
            self.optimizer.step()
            return loss.item()



      def GPFTrain(self, train_loader):
            self.prompt.train()
            total_loss = 0.0 
            for batch in train_loader:  
                  self.optimizer.zero_grad() 
                  batch = batch.to(self.device)
                  batch.x = self.prompt.add(batch.x)
                  out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = self.prompt_type)
                  out = self.answering(out)
                  loss = self.criterion(out, batch.y)  
                  loss.backward()  
                  self.optimizer.step()  
                  total_loss += loss.item()  
            return total_loss / len(train_loader) 



      def GPPTtrain(self, data):
            self.prompt.train()
            node_embedding = self.gnn(data.x, data.edge_index)
            out = self.prompt(node_embedding, data.edge_index)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
            loss = loss + 0.001 * constraint(self.device, self.prompt.get_TaskToken())
            self.pg_opi.zero_grad()
            loss.backward()
            self.pg_opi.step()
            mid_h = self.prompt.get_mid_h()
            self.prompt.update_StructureToken_weight(mid_h)
            return loss.item()



      def GpromptTrain(self, train_loader):
            self.prompt.train()
            total_loss = 0.0 
            accumulated_centers = None
            accumulated_counts = None
            for batch in train_loader:  
                  self.pg_opi.zero_grad() 
                  batch = batch.to(self.device)
                  out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = 'Gprompt')

                  # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
                  center, class_counts = center_embedding(out, batch.y, self.output_dim)
                   # 累积中心向量和样本数
                  if accumulated_centers is None:
                        accumulated_centers = center
                        accumulated_counts = class_counts
                  else:
                        accumulated_centers += center * class_counts
                        accumulated_counts += class_counts
                  criterion = Gprompt_tuning_loss()
                  loss = criterion(out, center, batch.y)  
                  loss.backward()  
                  self.pg_opi.step()  
                  total_loss += loss.item()
            # 计算加权平均中心向量
            mean_centers = accumulated_centers / accumulated_counts

            return total_loss / len(train_loader), mean_centers



      def RobustPromptInductiveTrain(self, train_loader, remaining_loader):
            #we update answering and prompt alternately.
            answer_epoch = 20  # 50 80
            prompt_epoch = 20  # 50 80
            
            # tune task head
            self.answering.train()
            self.prompt.eval()
            for epoch in range(1, answer_epoch + 1):
                  answer_loss = self.prompt.Tune(train_loader, remaining_loader, self.gnn,  self.answering, self.criterion, self.answer_opi, self.device)
                  print(("frozen gnn | frozen prompt | *tune answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, answer_loss)))

            # tune prompt
            self.answering.eval()
            self.prompt.train()
            for epoch in range(1, prompt_epoch + 1):
                  pg_loss = self.prompt.Tune( train_loader, remaining_loader, self.gnn, self.answering, self.criterion, self.pg_opi, self.device)
                  print(("frozen gnn | *tune prompt |frozen answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, pg_loss)))
            
            return pg_loss



      def RobustPromptTranductivetrain(self, data, iid_train, pruned_data, lambda_cmd, lambda_mse):
            self.prompt.train()
            self.optimizer.zero_grad() 
            # 这里要注意，和GPF不同，这里都是一个图而不是loader，所以一个epoch跑完后data.x就消失了，不能进行后向传播，要用一个新的值存储, 不能用data.x = self.prompt.add(data.x)，直接被覆盖，无法训练

            prompted_x = self.prompt.add(data.x)
            out       = self.gnn(prompted_x, data.edge_index, prompt = self.prompt, prompt_type = self.prompt_type)

            # sim 关注去噪
            out_clean = self.gnn(pruned_data.x, pruned_data.edge_index, prompt = self.prompt, prompt_type = self.prompt_type)
            # loss_mse = F.mse_loss(out[data.train_mask], out_clean[data.train_mask])
            # loss_mse = F.mse_loss(out, out_clean)

            # cmd 关注分布
            # loss_cmd = cmd(out[data.train_mask], out[iid_train, :])

            out = self.answering(out)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask]) # + lambda_cmd * loss_cmd + lambda_mse * loss_mse
            loss.backward()  
            self.optimizer.step()
            return loss




      def run(self):
            # for all-in-one and Gprompt we use k-hop subgraph
            if self.prompt_type in ['All-in-one', 'Gprompt', 'GPF', 'GPF-plus','RobustPrompt_I']:
                  train_loader = DataLoader(self.train_dataset, batch_size=100, shuffle=True)
                  test_loader = DataLoader(self.test_dataset, batch_size=100, shuffle=False)
                  val_loader = DataLoader(self.val_dataset, batch_size=100, shuffle=False)
                  if self.prompt_type == 'RobustPrompt_I':
                        print("Build a remaining dataloader.")
                        remaining_loader = DataLoader(self.remaining_dataset, batch_size=2500, shuffle=True)
                  
                  print("prepare induce graph data is finished!")

            if self.prompt_type == 'MultiGprompt':
                  # 使用预训练的GCN得到还没有加prompt的embs
                  embeds, _ = self.Preprompt.embed(self.features, self.sp_adj, True, None, False)
                  idx_train  = self.data.train_mask.nonzero().squeeze()
                  train_lbls = self.data.y[self.data.train_mask].type(torch.long) 

                  idx_val    = self.data.val_mask.nonzero().squeeze()
                  val_lbls   = self.data.y[self.data.val_mask].type(torch.long)

                  idx_test    = self.data.test_mask.nonzero().squeeze()
                  test_lbls   = self.data.y[self.data.test_mask].type(torch.long)

                  pretrain_embs = embeds[0, idx_train].type(torch.long)
                  val_embs      = embeds[0, idx_val].type(torch.long)
                  test_embs     = embeds[0, idx_test].type(torch.long)


            if self.prompt_type in ['RobustPrompt_T','RobustPrompt_Tplus']:
                  print("Prepare distribution nodes and prune the graph...")
                  ###############################################################
                  # 分布
                  # 选取测试集的节点
                  idx_train = self.data.train_mask.nonzero().squeeze()
                  all_idx = set(range(self.data.num_nodes)) - set(idx_train)
                  idx_test = torch.LongTensor(list(all_idx))
                  # method 1 : random select
                  # perm = torch.randperm(idx_test.shape[0])
                  # iid_train = idx_test[perm[:idx_train.shape[0]]] 
                  # method 2 : use all 
                  iid_train = idx_test
                  print("select distribution nodes done!")
                  ###############################################################

                  ###############################################################
                  # 噪声
                  # Prune edge index
                  edge_index = self.data.edge_index
                  cosine_sim = F.cosine_similarity(self.data.x[edge_index[0]], self.data.x[edge_index[1]])
                  # Define threshold t
                  threshold = 0.2
                  # Identify edges to keep
                  keep_edges = cosine_sim >= threshold
                  # Filter edge_index to only keep edges above the threshold
                  pruned_edge_index = edge_index[:, keep_edges]
                  pruned_data       = Data(x=self.data.x, edge_index=pruned_edge_index)
                  print(pruned_data)
                  print(self.data)
                  print("prune the graph done!")
                  ###############################################################
                  # 对两个loss控制的超参数
                  lambda_cmd = 0.5
                  lambda_mse = 0.5
                  ###############################################################
                  # 打印一下节点周围邻居节点的标签，研究一下


            print("run {}".format(self.prompt_type))
            best_val_acc = final_test_acc = 0
            for epoch in range(0, self.epochs):
            # 用tqdm 更简洁
            # pbar = tqdm(range(0, self.epochs))
            # for epoch in pbar:
                  t0 = time.time()
                  if self.prompt_type == 'None':
                        loss = self.train(self.data)
                        print("Train Done!")
                        val_acc = GNNNodeEva(self.data, self.data.val_mask, self.gnn, self.answering)
                        print("Val Done!")
                        test_acc = GNNNodeEva(self.data, self.data.test_mask, self.gnn, self.answering)
                        print("Test Done!")
                        
                  elif self.prompt_type == 'All-in-one':
                        # print("run All-in-one Prompt")
                        loss = self.AllInOneTrain(train_loader)
                        # 看下训练集的训练情况，是不是在被攻击数据上过拟合了 不用的话就注释掉
                        train_acc, F1  = AllInOneEva(train_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                        print("train batch Done!")
                        val_acc, F1    = AllInOneEva(val_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                        print("val batch Done!")
                        test_acc, F1   = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                        print("test batch Done!")
                  
                  elif self.prompt_type == 'GPPT':
                        print("run GPPT Prompt")
                        loss     = self.GPPTtrain(self.data)
                        train_acc,  F1  = GPPTEva(self.data, self.data.train_mask, self.gnn, self.prompt, self.output_dim, self.device)
                        print("Train Done!")
                        val_acc,  F1  = GPPTEva(self.data, self.data.val_mask, self.gnn, self.prompt, self.output_dim, self.device)
                        print("Val Done!")
                        test_acc, F1  = GPPTEva(self.data, self.data.test_mask, self.gnn, self.prompt, self.output_dim, self.device)
                        print("Test Done!")

                  elif self.prompt_type =='Gprompt':
                        print("run Graph Prompt")
                        loss, center =  self.GpromptTrain(train_loader)
                        train_acc, F1 = GpromptEva(train_loader, self.gnn, self.prompt, center, self.output_dim, self.device)
                        print("train batch Done!")

                        val_acc, F1 = GpromptEva(val_loader, self.gnn, self.prompt, center, self.output_dim, self.device)
                        print("val batch Done!")

                        test_acc, F1= GpromptEva(test_loader, self.gnn, self.prompt, center, self.output_dim, self.device)
                        print("test batch Done!")


                  elif self.prompt_type in ['GPF', 'GPF-plus']:
                        print("run GPF/GPF-Plus Prompt")
                        loss = self.GPFTrain(train_loader)
                        train_acc, F1 = GPFEva(train_loader, self.gnn, self.prompt, self.answering, self.output_dim, self.device)    
                        # print("train batch Done!")


                        val_acc, F1 = GPFEva(val_loader, self.gnn, self.prompt, self.answering, self.output_dim, self.device)    
                        # print("val batch Done!")

                        test_acc, F1 = GPFEva(test_loader, self.gnn, self.prompt, self.answering, self.output_dim, self.device)    
                        # print("test batch Done!")


                  elif self.prompt_type == 'MultiGprompt':
                        print("run MultiGprompt")
                        loss = self.MultiGpromptTrain(pretrain_embs, train_lbls, idx_train)
   
                        # 记得 open prompt
                        prompt_feature = self.feature_prompt(self.features)
                        train_acc, F1 = MultiGpromptEva(pretrain_embs, train_lbls, idx_train, prompt_feature, self.Preprompt, self.DownPrompt, self.sp_adj, self.output_dim, self.device)
                        print("Train Done!")

                        # 记得 open prompt
                        prompt_feature = self.feature_prompt(self.features)
                        val_acc, F1 = MultiGpromptEva(val_embs, val_lbls, idx_val, prompt_feature, self.Preprompt, self.DownPrompt, self.sp_adj, self.output_dim, self.device)
                        print("Val Done!")

                        # 记得 open prompt
                        prompt_feature = self.feature_prompt(self.features)
                        test_acc, F1 = MultiGpromptEva(test_embs, test_lbls, idx_test, prompt_feature, self.Preprompt, self.DownPrompt, self.sp_adj, self.output_dim, self.device)
                        print("Test Done!")



                  # 图形状的prompt
                  elif self.prompt_type == 'RobustPrompt_I':
                        print("run RobustPrompt_I")
                        loss = self.RobustPromptInductiveTrain(train_loader, remaining_loader)
                        # 看下训练集的训练情况，是不是在被攻击数据上过拟合了 不用的话就注释掉
                        train_acc, F1  = RobustPromptInductiveEva(train_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                        print("RobustPrompt_I train batch Done!")
                        val_acc, F1    = RobustPromptInductiveEva(val_loader,   self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                        print("RobustPrompt_I val batch Done!")
                        test_acc, F1   = RobustPromptInductiveEva(test_loader,  self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                        print("RobustPrompt_I test batch Done!")


                  elif self.prompt_type in ['RobustPrompt_T', 'RobustPrompt_Tplus']:
                        # print("run RobustPrompt_T/RobustPrompt_TPlus Prompt")
                        loss            = self.RobustPromptTranductivetrain(self.data, iid_train, pruned_data, lambda_cmd, lambda_mse)
                        train_acc,  F1  = RobustPromptTranductiveEva(self.data, self.data.train_mask, self.gnn, self.prompt, self.answering, self.output_dim, self.device)
                        # print("Train Done!")
                        val_acc,  F1    = RobustPromptTranductiveEva(self.data, self.data.val_mask,   self.gnn, self.prompt, self.answering, self.output_dim, self.device)
                        # print("Val Done!")
                        test_acc, F1    = RobustPromptTranductiveEva(self.data, self.data.test_mask,  self.gnn, self.prompt, self.answering, self.output_dim, self.device)
                        # print("Test Done!")


                  if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        final_test_acc = test_acc
                  # print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f} | val Accuracy {:.4f} | test Accuracy {:.4f} ".format(epoch + 1, time.time() - t0, loss, val_acc, test_acc)) 
                  
                  # 看下训练集的训练情况，是不是在被攻击数据上过拟合了
                  # 果然 Epoch 009 |  Time(s) 5.1142 | Loss 3.3146 | train Accuracy 0.7143 | val Accuracy 0.3429 | test Accuracy 0.3059
                  print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f} | train Accuracy {:.4f} | val Accuracy {:.4f} | test Accuracy {:.4f} ".format(epoch + 1, time.time() - t0, loss, train_acc, val_acc, test_acc))       

                  # 使用tqdm进行显示 更简洁
                  # pbar.set_description("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f} | train Accuracy {:.4f} | val Accuracy {:.4f} | test Accuracy {:.4f} ".format(epoch + 1, time.time() - t0, loss, train_acc, val_acc, test_acc))       



            print(f'Final Test: {final_test_acc:.4f}')
            print("Node Task completed")

            return final_test_acc.cpu().numpy() if isinstance(final_test_acc, torch.Tensor) else final_test_acc

