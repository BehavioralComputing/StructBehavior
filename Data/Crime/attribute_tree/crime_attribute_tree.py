import torch
import numpy as np
import pandas as pd
import json
import os
from transformers import pipeline, AutoModel, AutoTokenizer, BertTokenizer, BertModel
from datetime import datetime as dt
from torch.utils.data import Dataset
from tqdm import tqdm
import tree
from crime_max1SE_new import get_weight, knn_maxE1, get_adj_matrix, add_knn
from joblib import Parallel, delayed
import time

device = "cuda:0"
PWD = os.path.dirname(os.path.realpath(__file__))

raw_data_path = PWD + "/../preprocessed_data"
pt_path = PWD + "/../material_data"
save_path = PWD + "/mid"
pretrained_model_path = PWD + "/../../../bert/bert-base-uncased"

if not os.path.exists(save_path):
    os.makedirs(save_path)


class Crime_attribute_tree(Dataset):
    def __init__(self, device="cpu", process=True, save=True):
        self.raw_data_path = raw_data_path
        self.save_path = save_path
        self.pretrained_model_path = pretrained_model_path
        self.device = device
        if process:
            print("---Loading Raw Csv Files...---")
            print("Loading crime labeled.csv")
            self.df_data_labeled = pd.read_csv(raw_data_path + "/crime_labeled.csv")

            self.df_data_labeled = self.df_data_labeled.drop(columns=["Label"])
            self.df_label = torch.load(pt_path + "/crime_label.pt")
            print("---Loading Raw Csv Files Finished!---")
            self.save = save

    def attribute_graph_generate(self):
        print("---Generating Attribute Graph...---")
        attributes_2_index_path = self.save_path + "/attributes_2_index.json"
        index_2_attributes_path = self.save_path + "/index_2_attributes.json"
        path_nodes_feature = self.save_path + "/attribute_nodes_feature.pt"
        path_edge_index = self.save_path + "/edge_index.pt"

        if not os.path.exists(attributes_2_index_path):
            attributes = []

            for attribute in self.df_data_labeled.columns:
                attributes.append(attribute.strip())

            attributes_dict = {val: index for index, val in enumerate(attributes)}

            with open(attributes_2_index_path, "w") as file:
                json.dump(attributes_dict, file)
                print("attributes_2_index.json file created!")
        else:
            print("attributes_2_index.json file existed!")

        if not os.path.exists(index_2_attributes_path):
            attributes = []
            for attribute in self.df_data_labeled.columns:
                attributes.append(attribute.strip())

            attributes_dict = {index: val for index, val in enumerate(attributes)}

            with open(index_2_attributes_path, "w") as file:
                json.dump(attributes_dict, file)
                print("index_2_attributes.json file created!")
        else:
            print("index_2_attributes.json file existed!")

        if not os.path.exists(path_nodes_feature):
            print("nodes feature generating...")
            start_time = time.time()

            graphs_list = Parallel(n_jobs=2)(
                delayed(self.attribute_feature_extract)(row)
                for row in tqdm(self.df_data_labeled.itertuples(index=False))
            )
            graphs_tensor = torch.stack(graphs_list).to("cpu")
            end_time = time.time()
            print("time of parallel process: ", end_time - start_time)
            print("attribute_nodes_feature: " + str(graphs_tensor.shape))
            if self.save:
                torch.save(graphs_tensor, path_nodes_feature)
                print("attribute_nodes_feature.pt file created!")
        else:
            print("attribute_nodes_feature.pt file existed!")

        if not os.path.exists(path_edge_index):
            print("edges index generating...")
            with open(attributes_2_index_path, "r") as file:
                attributes_2_index = json.load(file)

            graph_edge_index_tensor = self.single_graph_edge_index(attributes_2_index)
            graph_edge_index_tensor = graph_edge_index_tensor.unsqueeze(0)
            graphs_edge_index_tensor = graph_edge_index_tensor.repeat(
                len(self.df_data_labeled), 1, 1
            ).to("cpu")
            print("edge_index: " + str(graphs_edge_index_tensor.shape))
            if self.save:
                torch.save(graphs_edge_index_tensor, path_edge_index)
                print("edge_index.pt file created!")
        else:
            print("edge_index.pt file existed!")
        print("---Attribute Graph Generated!---")

    def attribute_feature_extract(self, row):

        graph_tensor = []
        tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
        model = BertModel.from_pretrained(self.pretrained_model_path)
        model.to(device)
        graph_list = []
        none_str = "None"
        none_text = tokenizer(none_str, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**none_text)
            none_embedding_pool = outputs.last_hidden_state.mean(dim=1).to(device)
            none_embedding = none_embedding_pool[0]

        for i, value in enumerate(row):

            if pd.isna(value):
                print(f"Attribute {i} is NaN, using None embedding.")
                graph_list.append(none_embedding)
                continue
            else:
                encoded_text = tokenizer(
                    str(value), truncation=True, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    outputs = model(**encoded_text)
                    value_embedding_pool = outputs.last_hidden_state.mean(dim=1).to(
                        device
                    )
                    graph_list.append(value_embedding_pool[0])

        assert len(graph_list) == 19
        graph_tensor = torch.stack(graph_list).to(device)
        return graph_tensor

    def single_graph_edge_index(self, attributes_2_index):

        tim_pla_peo_weap = [
            "Vict Age",
            "Vict Sex",
            "Vict Descent",
            "Premis Desc",
            "Weapon Desc",
            "Month_OCC",
            "Day_OCC",
            "Year_OCC",
            "Hour",
            "Minute",
        ]

        pcl = ["Premis Desc", "Cross Street"]

        rpa = ["Rpt Dist No", "Premis Desc", "AREA NAME"]

        rsp = ["Status", "Part 1-2", "Rpt Dist No"]

        rdp = ["Day_Rptd", "Rpt Dist No", "Premis Desc"]

        dym = ["Month_Rptd", "Day_Rptd", "Year_Rptd"]

        ddd = ["Date Difference", "Day_Rptd", "Day_OCC"]

        complete_link = [tim_pla_peo_weap, pcl, rpa, rsp, rdp, dym, ddd]

        single_link = []

        graph_edge_index = []

        for complete_part in complete_link:
            for i in range(len(complete_part)):
                for j in range(i + 1, len(complete_part)):
                    graph_edge_index.append(
                        [
                            attributes_2_index[complete_part[i]],
                            attributes_2_index[complete_part[j]],
                        ]
                    )

        for i in single_link:
            graph_edge_index.append(
                [attributes_2_index[i[0]], attributes_2_index[i[1]]]
            )
        graph_edge_index = (
            torch.tensor(graph_edge_index, dtype=torch.long)
            .t()
            .contiguous()
            .to(self.device)
        )
        return graph_edge_index

    def attribute_tree_generate(self):
        print("---Generating Attribute Tree...---")
        dataset = "crime"
        for depth in range(3, 9):
            tree.load_attribute_coding_tree(dataset, depth)
        print("---Attribute Tree Generated!---")

    def train_val_test_mask(self):
        raise NotImplementedError(
            "当前的结点划分不正确，需要重新实现 train_val_test_mask 方法。"
        )
        train_idx = range(8278)
        val_idx = range(8278, 8278 + 2365)
        test_idx = range(8278 + 2365, 8278 + 2365 + 1183)
        return train_idx, val_idx, test_idx

    def dataloader(self):
        edge_index, edge_type = self.Build_Graph()
        train_idx, val_idx, test_idx = self.train_val_test_mask()
        return edge_index, edge_type, train_idx, val_idx, test_idx


if __name__ == "__main__":
    dataset = Crime_attribute_tree(device=device, process=True, save=True)
    dataset.attribute_graph_generate()
    dataset.attribute_tree_generate()
