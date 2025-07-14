from attribute_tree.sep_g import SEP_G
from backbone.rgcn import FACNConv as RGCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
from torch_geometric.data import Data
import argparse
import pickle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import pandas as pd

PWD = os.path.dirname(os.path.realpath(__file__))

def edge_mask(edge_index, edge_attr, pe):
    edge_index = edge_index.clone()
    edge_num = edge_index.shape[1]
    pre_index = torch.bernoulli(torch.ones(edge_num) * pe) == 0
    pre_index.to(edge_index.device)
    edge_index = edge_index[:, pre_index]
    edge_attr = edge_attr.clone()
    edge_attr = edge_attr[pre_index]
    return edge_index, edge_attr

def relational_undirected(edge_index, edge_type):
    device = edge_index.device
    relation_num = edge_type.max() + 1
    edge_index = edge_index.clone()
    edge_type = edge_type.clone()
    r_edge = []
    for i in range(relation_num):
        e1 = edge_index[:, edge_type == i].unique(dim=1)
        e2 = e1.flip(0)
        edges = torch.cat((e1, e2), dim=1)
        r_edge.append(edges)
    edge_type = torch.cat(
        [torch.tensor([i] * e.shape[1]) for i, e in enumerate(r_edge)],
        dim=0).to(device)
    edge_index = torch.cat(r_edge, dim=1)

    return edge_index, edge_type

class Attribute_Layer(nn.Module):
    def __init__(self, args):
        super(Attribute_Layer, self).__init__()
        self.args = args
        self.sep_g = SEP_G(self.args)
    
    def forward(self, data):
        x = self.sep_g(data)
        return x
    
    
class RGCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.rel_names = args.rel_names  
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(args.input_dim, args.input_dim)
            for rel in self.rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(args.input_dim, args.input_dim)
            for rel in self.rel_names}, aggregate='sum')

    def forward(self, graph, node_feats):
        edge_weight_dict = {
            rel: graph.edges[rel].data['weight_power']  
            for rel in self.rel_names
        }
        
        h = self.conv1(
            graph, node_feats,
            mod_kwargs={
                rel: {'edge_weight': edge_weight_dict[rel]}
                for rel in self.rel_names
            }
        )
        h = {k: F.relu(v) for k, v in h.items()}
        
        h = self.conv2(
            graph, h,
            mod_kwargs={
                rel: {'edge_weight': edge_weight_dict[rel]}
                for rel in self.rel_names
            }
        )
        return h

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.input_dim, args.input_dim),
            nn.ReLU(),
            nn.Linear(args.input_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )

    def forward(self, x):
        return self.layers(x)

# arXiv Structure RGCN
class normal_RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names
        }, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names
        }, aggregate='sum')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

class AttributePool(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_linear = nn.Linear(args.feature_num, args.input_dim)
        self.rgcn = RGCN(args)
        self.mlp = MLP(args)
        self.device = args.device
        self.hidden_dim = args.input_dim
        self.map_data = args.map_data

    def forward(self, g: dgl.DGLGraph):
        h = g.ndata['feat']  
        h = {k: F.relu(self.input_linear(v)) for k, v in h.items()}  
        h = self.rgcn(g, h)

        columns = list(self.map_data.columns)
        index_matrix = torch.full((args.node_num, len(columns)), -1, dtype=torch.long)
        for i, col in enumerate(columns):
            col_values = self.map_data[col].values[:args.node_num]
            index_matrix[:, i] = torch.tensor([
                int(v) if pd.notna(v) else -1 for v in col_values
            ], dtype=torch.long)

        
        all_feat_sum = torch.zeros(args.node_num, self.hidden_dim, device=self.device)
        valid_counts = torch.zeros(args.node_num, device=self.device)
        
        for i, col in enumerate(columns):
            idx = index_matrix[:, i]  
            valid_mask = idx != -1  
            selected_feats = h[col][idx.clamp(min=0)]  
            selected_feats[~valid_mask] = 0  
            all_feat_sum += selected_feats
            valid_counts += valid_mask.float().to(self.device)
        
        valid_counts[valid_counts == 0] = 1
        all_mean_feat = all_feat_sum / valid_counts.unsqueeze(1)
        return self.mlp(all_mean_feat)


class StructBehavior(nn.Module):
    def __init__(self, args, idx):
        super(StructBehavior, self).__init__()
        self.args = args
        self.idx = idx
        # Layer3
        self.attribute_layer = Attribute_Layer(self.args)
        # Layer2
        self.attribute_pool_layer = AttributePool(self.args)
        # Layer1
        self.backbone = normal_RGCN(
            in_feats = self.args.in_feats,
            hid_feats = self.args.rgcn_hidden_feats, 
            out_feats = self.args.hidden_dim,
            rel_names = self.args.rgcn_rel_names
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.args.hidden_dim * 3, self.args.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.args.hidden_dim, self.args.num_classes)
        )
        self.proj_1_2 = nn.Sequential(
            nn.Linear(self.args.hidden_dim,
                      self.args.proj_dim), nn.LeakyReLU(),
            nn.Linear(self.args.proj_dim, self.args.hidden_dim)
        )
        self.proj_1_3 = nn.Sequential(
            nn.Linear(self.args.hidden_dim,
                      self.args.proj_dim), nn.LeakyReLU(),
            nn.Linear(self.args.proj_dim, self.args.hidden_dim)
        )
        self.proj_2_3 = nn.Sequential(
            nn.Linear(self.args.hidden_dim,
                      self.args.proj_dim), nn.LeakyReLU(),
            nn.Linear(self.args.proj_dim, self.args.hidden_dim)
        )
        self.test_results = []
    
    # https://blog.csdn.net/weixin_44966641/article/details/120382198
    def infonce_loss(self,
                     emb_i,
                     emb_j,
                     temperature=0.1):  
        batch_size = emb_i.shape[0]
        negatives_mask = (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(
                self.args.device).float()  # (2*bs, 2*bs)
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = torch.mm(representations, representations.t())

        sim_ij = torch.diag(similarity_matrix, batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / temperature)  # 2*bs
        denominator = negatives_mask * torch.exp(
            similarity_matrix / temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(
            nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss
    
    def forward(self, batch):
        out_1 = self.backbone(batch['data']['graph'], batch['data']['inputs'])
        out_1 = out_1['node']
        out_1 = out_1[:self.args.node_num, :]

        out_2 = self.attribute_pool_layer(batch['graph'])

        out_3 = self.attribute_layer(batch)

        loss_1_2 = self.infonce_loss(
            self.proj_1_2(out_1), self.proj_1_2(out_2), self.args.temperature
        )
        loss_2_3 = self.infonce_loss(
            self.proj_2_3(out_2), self.proj_2_3(out_3), self.args.temperature
        )

        if self.training:
            # Training
            train_out = torch.cat([out_1, out_2, out_3], dim = 1)[self.idx['train_idx']]
            train_out = self.classifier(train_out)
            loss_ce = F.cross_entropy(train_out, self.idx['train_label'])
            loss = loss_ce + loss_1_2 * self.args.alpha_1_2 + loss_2_3 * self.args.alpha_2_3
            return loss
        else:
            # Validation
            val_out = torch.cat([out_1, out_2, out_3], dim = 1)[self.idx['val_idx']]
            val_out = self.classifier(val_out)
            val_loss = F.cross_entropy(val_out, self.idx['val_label'])
            val_acc = accuracy_score(
                self.idx['val_label'].cpu().numpy(),
                torch.argmax(val_out, dim=1).cpu().numpy()
            )

            # Test
            test_out = torch.cat([out_1, out_2, out_3], dim = 1)[self.idx['test_idx']]
            test_out = self.classifier(test_out)
            test_label = self.idx['test_label'].cpu().numpy()
            test_pred = torch.argmax(test_out, dim=1).cpu().numpy()

            test_acc = accuracy_score(test_label, test_pred)
            test_f1 = f1_score(test_label, test_pred, average='macro')
            test_recall = recall_score(test_label, test_pred, average='macro')
            test_precision = precision_score(test_label, test_pred, average='macro')

            self.test_results.append(
                [test_acc, test_f1, test_recall, test_precision])
            return val_acc, val_loss.item(), test_acc, test_precision, test_recall, test_f1

    def get_test_results(self):
        return self.test_results

def create_dgl_hetero_graph(edge_index, edge_type, num_nodes):
    rel_names = [f'rel_{i}' for i in range(edge_type.max().item() + 1)]
    
    edge_dict = {}
    for i, rel_name in enumerate(rel_names):
        mask = edge_type == i
        if mask.sum() > 0:
            src = edge_index[0][mask]
            dst = edge_index[1][mask]
            edge_dict[('node', rel_name, 'node')] = (src, dst)
    
    for i, rel_name in enumerate(rel_names):
        if ('node', rel_name, 'node') not in edge_dict:
            edge_dict[('node', rel_name, 'node')] = (torch.tensor([]), torch.tensor([]))
    
    graph = dgl.heterograph(edge_dict, num_nodes_dict={'node': num_nodes})
    return graph, rel_names

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        # Random Seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.args = args
        self.load_data()  
        self.model = StructBehavior(self.args, self.idx).to(self.args.device)
        self.save_top_k = args.save_top_k
        self.patience = 0
        self.best_loss_epoch = 0
        self.best_acc_epoch = 0
        self.best_loss = 1e9
        self.best_loss_acc = -1e9
        self.best_acc = -1e9
        self.best_acc_loss = 1e9
        self.test_results = []

    def load_data(self):
        # Layer3 Data
        attri_raw_path = os.path.join(PWD, 'Data/Crime/attribute_tree/mid')
        tree_path = os.path.join(
            PWD, 'Data/Crime/attribute_tree/trees',
            '%s_%s.pickle' % (self.args.dataset, self.args.tree_depth)
        )
        
        with open(os.path.join(attri_raw_path, 'knn_graphs_edge_index.pickle'), 'rb') as fp2:
            self.knn_edge_index_list = pickle.load(fp2)

        
        with open(tree_path, 'rb') as fp1:
            self.layer_data = pickle.load(fp1)
        self.args.num_features = self.args.hidden_dim
        
        self.x_list = torch.load(os.path.join(attri_raw_path, 'attribute_nodes_feature.pt'))

        # Layer2 Data
        attribute_pool_graph_path = os.path.join(PWD, 'Data/Crime/attribute_pool/crime_weighted_graph.dgl')
        loaded_graphs, _ = dgl.load_graphs(attribute_pool_graph_path)
        loaded_graphs = loaded_graphs[0]  
        self.graph = loaded_graphs.to(self.args.device)
        args.rel_names = self.graph.etypes

        map_data_path = os.path.join(PWD, 'Data/Crime/attribute_pool/crime_data_df.csv')
        map_data = pd.read_csv(map_data_path)
        args.map_data = map_data  

        # Layer1 Data
        whole_raw_path = os.path.join(PWD, 'Data/Crime/material_data')
        label = torch.load(os.path.join(whole_raw_path, 'crime_label.pt'))
        x = torch.cat([
            torch.load(os.path.join(whole_raw_path, 'crime_num_properties_tensor.pt')),
            torch.load(os.path.join(whole_raw_path, 'crime_cat_properties_tensor.pt')),
        ], dim = 1)

        edge_index = torch.load(os.path.join(whole_raw_path, 'crime_edge_index.pt'))
        edge_type = torch.load(os.path.join(whole_raw_path, 'crime_edge_type.pt'))

        edge_index, edge_type = relational_undirected(edge_index, edge_type)
        self.args.num_relations = edge_type.max() + 1

        num_nodes = x.shape[0]
        graph, rgcn_rel_names = create_dgl_hetero_graph(edge_index, edge_type, num_nodes)
        graph = graph.to(self.args.device)

        self.args.rgcn_rel_names = rgcn_rel_names
        self.args.in_feats = x.shape[1]
        self.args.rgcn_hidden_feats = 128

        inputs = {'node': x.to(self.args.device)}
        
        self.data = {
            'graph': graph, 
            'inputs': inputs
        }

        # self.data = data

        train_val_test_idx = np.arange(self.args.node_num)
        train_idx, test_idx = train_test_split(train_val_test_idx, test_size=0.3, random_state=self.args.seed)
        val_idx, test_idx = train_test_split(test_idx, test_size=0.3333, random_state=self.args.seed)  

        train_label = label[train_idx].to(self.args.device)
        val_label = label[val_idx].to(self.args.device)
        test_label = label[test_idx].to(self.args.device)

        self.idx = {
            'train_idx': train_idx, 
            'val_idx': val_idx, 
            'test_idx': test_idx, 
            'train_label': train_label, 
            'val_label': val_label, 
            'test_label': test_label
        }

    def organize_val_log(self, val_loss, val_acc, epoch):
        if val_loss < self.best_loss:
            self.best_loss_acc = val_acc
            self.best_loss = val_loss
            self.best_loss_epoch = epoch
            self.patience = 0
        else:
            self.patience += 1

        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_acc_loss = val_loss
            self.best_acc_epoch = epoch

    def train(self):
        best_test_acc = 0.0

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr = self.args.lr,
            weight_decay = self.args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max = 16, 
            eta_min = 0
        )
        val_accs = []
        val_losses = []
        test_accs = []

        for epoch in range(self.args.epochs):
            self.model.train()
            batch = {
                'data': self.data, # layer_1
                'graph': self.graph, # layer_2
                'x_list': self.x_list.to(self.args.device), # layer_3
                'knn_edge_index_list': self.knn_edge_index_list, # layer_3
                'layer_data': self.layer_data # layer_3
            }
            self.optimizer.zero_grad()
            loss = self.model(batch)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Validation
            val_acc, val_loss, test_acc, test_precision, test_recall, test_f1 = self.eval(batch)
            print(
                'epoch: %d, val_acc: %.4f, val_loss: %.4f, test_acc: %.4f, test_precision: %.4f, test_recall: %.4f, test_f1: %.4f'
                % (epoch, val_acc, val_loss, test_acc, test_precision, test_recall, test_f1)
            )

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            self.organize_val_log(val_loss, val_acc, epoch)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            test_accs.append(test_acc)

        return best_test_acc

    def eval(self, batch):
        self.model.eval()
        return self.model(batch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'SEP')
    parser.add_argument('--dataset', type = str, default = 'crime')
    parser.add_argument('--node_num', type = int, default = 5553)
    parser.add_argument('--attribute_num', type = int, default = 19)
    parser.add_argument('--num_classes', type = int, default = 10)

    parser.add_argument('--feature_num', type = int, default = 768)
    parser.add_argument('--prop_num', type=int, default=2)
    parser.add_argument('--cat_num', type=int, default=139)

    parser.add_argument('--pe', type=float, default=0.2)  
    parser.add_argument('--pf', type=float, default=0.2)  

    parser.add_argument('--epochs', default = 500, type = int)
    parser.add_argument('--tree_depth', type = int, default = 4)
    parser.add_argument('--conv', type = str, default = 'GCN')
    parser.add_argument('--input_dim', type = int, default = 128)
    parser.add_argument('--hidden_dim', type = int, default = 32)
    parser.add_argument('--proj_dim', type=int, default=16)
    parser.add_argument("--gpu", type = int, default = 0)  
    parser.add_argument('--patience', type = int, default = 80)
    parser.add_argument('--save_top_k', type = int, default = 6)  

    parser.add_argument('--alpha_1_2', type = float, default = 0.12)
    parser.add_argument('--alpha_2_3', type=float, default = 0.1)

    parser.add_argument('--temperature', type = float, default = 0.1)
    parser.add_argument('--seed', type = int, default = 42, help = 'seed')
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--weight_decay', type = float, default = 3e-3)
    parser.add_argument("--dropout", type = float, default = 0.5)
    parser.add_argument('--conv_dropout', type = float, default = 0.5)
    parser.add_argument('--pooling_dropout', type = float, default = 0.5)
    parser.add_argument('-gp',
                        '--global-pooling',
                        type = str,
                        default = "average",
                        choices = ["sum", "average"],
                        help = 'Pooling for over nodes: sum or average')

    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    trainer = Trainer(args)
    test_acc = trainer.train()
    print('test_acc', test_acc)
