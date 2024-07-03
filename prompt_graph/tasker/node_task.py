import torch
from torch_geometric.loader import DataLoader
from prompt_graph.utils import constraint,  center_embedding, Gprompt_tuning_loss
from prompt_graph.evaluation import GPPTEva, GNNNodeEva, GpromptEva, MultiGpromptEva, GPFEva
from prompt_graph.utils import process
from .task import BaseTask
import time
import warnings
from prompt_graph.data import induced_graphs, split_induced_graphs,load4node_shot_index
from prompt_graph.evaluation import AllInOneEva
import pickle
import os
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix

warnings.filterwarnings("ignore")

class NodeTask(BaseTask):
      def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.task_type = 'NodeTask'
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
            print('a',self.output_dim)
            features, _ = process.preprocess_features(features)
            self.sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj).to(self.device)
            self.labels = torch.FloatTensor(labels[np.newaxis])
            self.features = torch.FloatTensor(features[np.newaxis]).to(self.device)
            # print("labels",labels)
            print("adj",self.sp_adj.shape)
            print("feature",features.shape)



      def load_data(self):

            self.data, self.dataset = load4node_shot_index(self.dataset_name, preprocess_method = self.preprocess_method, shot_num = self.shot_num, run_split= self.run_split)

            if self.prompt_type == 'MultiGprompt':
                  self.process_multigprompt_data(self.data)
            else:
                  self.input_dim = self.data.x.shape[1]
                  self.output_dim = self.dataset.num_classes


            if self.prompt_type in ['All-in-one','Gprompt', 'GPF', 'GPF-plus']:
                  file_dir = './data/{}/induced_graph/shot_{}/{}'.format(self.dataset_name, str(self.shot_num), str(self.run_split))
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
            
            answer_epoch = 80  # 50
            prompt_epoch = 80  # 50
            
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
            logits = self.DownPrompt(pretrain_embs,pretrain_embs1, train_lbls,1).float().to(self.device)
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









      def run(self):
            # for all-in-one and Gprompt we use k-hop subgraph
            if self.prompt_type in ['All-in-one', 'Gprompt', 'GPF', 'GPF-plus']:
                  train_loader = DataLoader(self.train_dataset, batch_size=100, shuffle=True)
                  test_loader = DataLoader(self.test_dataset, batch_size=100, shuffle=False)
                  val_loader = DataLoader(self.val_dataset, batch_size=100, shuffle=False)
                  print("prepare induce graph data is finished!")

            if self.prompt_type == 'MultiGprompt':
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



            best_val_acc = final_test_acc = 0
            for epoch in range(0, self.epochs):
                  t0 = time.time()
                  if self.prompt_type == 'None':
                        loss = self.train(self.data)
                        print("Train Done!")
                        val_acc = GNNNodeEva(self.data, self.data.val_mask, self.gnn, self.answering)
                        print("Val Done!")
                        test_acc = GNNNodeEva(self.data, self.data.test_mask, self.gnn, self.answering)
                        print("Test Done!")
                        
                  elif self.prompt_type == 'All-in-one':
                        print("run All-in-one Prompt")
                        loss = self.AllInOneTrain(train_loader)
                        print("train batch Done!")
                        val_acc, F1  = AllInOneEva(val_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                        print("val batch Done!")
                        test_acc, F1 = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                        print("test batch Done!")
                  
                  elif self.prompt_type == 'GPPT':
                        print("run GPPT Prompt")
                        loss     = self.GPPTtrain(self.data)
                        print("Train Done!")
                        val_acc,  F1  = GPPTEva(self.data, self.data.val_mask, self.gnn, self.prompt, self.output_dim, self.device)
                        print("Val Done!")
                        test_acc, F1  = GPPTEva(self.data, self.data.test_mask, self.gnn, self.prompt, self.output_dim, self.device)
                        print("Test Done!")

                  elif self.prompt_type =='Gprompt':
                        print("run Graph Prompt")
                        loss, center =  self.GpromptTrain(train_loader)
                        print("train batch Done!")
                        val_acc, F1 = GpromptEva(val_loader, self.gnn, self.prompt, center, self.output_dim, self.device)
                        print("val batch Done!")
                        test_acc, F1= GpromptEva(test_loader, self.gnn, self.prompt, center, self.output_dim, self.device)
                        print("test batch Done!")


                  elif self.prompt_type in ['GPF', 'GPF-plus']:
                        print("run GPF/GPF-Plus Prompt")
                        loss = self.GPFTrain(train_loader)
                        print("train batch Done!")

                        val_acc, F1 = GPFEva(val_loader, self.gnn, self.prompt, self.answering, self.output_dim, self.device)    
                        print("val batch Done!")

                        test_acc, F1 = GPFEva(test_loader, self.gnn, self.prompt, self.answering, self.output_dim, self.device)    
                        print("test batch Done!")




                  elif self.prompt_type == 'MultiGprompt':
                        print("run MultiGprompt")
                        loss = self.MultiGpromptTrain(pretrain_embs, train_lbls, idx_train)
                        print("Train Done!")

                        # 记得 open prompt
                        prompt_feature = self.feature_prompt(self.features)
                        val_acc, F1 = MultiGpromptEva(val_embs, val_lbls, idx_val, prompt_feature, self.Preprompt, self.DownPrompt, self.sp_adj, self.output_dim, self.device)
                        print("Val Done!")

                        # 记得 open prompt
                        prompt_feature = self.feature_prompt(self.features)
                        test_acc, F1 = MultiGpromptEva(test_embs, test_lbls, idx_test, prompt_feature, self.Preprompt, self.DownPrompt, self.sp_adj, self.output_dim, self.device)
                        print("Test Done!")



                  if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        final_test_acc = test_acc
                  print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f} | val Accuracy {:.4f} | test Accuracy {:.4f} ".format(epoch + 1, time.time() - t0, loss, val_acc, test_acc)) 

            print(f'Final Test: {final_test_acc:.4f}')
            print("Node Task completed")

            return final_test_acc.cpu().numpy() if isinstance(final_test_acc, torch.Tensor) else final_test_acc

