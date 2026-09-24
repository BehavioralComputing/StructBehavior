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
from max1SE import get_weight, knn_maxE1, get_adj_matrix, add_knn
from joblib import Parallel, delayed
import time

device = "cuda:0"
PWD = os.path.dirname(os.path.realpath(__file__))

raw_data_path = PWD + "/../raw"
save_path = PWD + "/mid"
pretrained_model_path = PWD + "/../../../bert/bert-base-uncased"

if not os.path.exists(save_path):
    os.makedirs(save_path)


class Twibot20_attribute_tree(Dataset):
    def __init__(self, device="cpu", process=True, save=True):
        self.raw_data_path = raw_data_path
        self.save_path = save_path
        self.pretrained_model_path = pretrained_model_path
        self.device = device
        if process:
            print("---Loading Raw Json Files...---")
            print("loading train.json")
            df_train = pd.read_json(raw_data_path + "/train.json")
            print("loading dev.json")
            df_dev = pd.read_json(raw_data_path + "/dev.json")
            print("loading test.json")
            df_test = pd.read_json(raw_data_path + "/test.json")
            df_train_label = df_train.iloc[:, [5]]
            df_dev_label = df_dev.iloc[:, [5]]
            df_test_label = df_test.iloc[:, [5]]
            df_label = pd.concat(
                [df_train_label, df_dev_label, df_test_label], ignore_index=True
            )
            self.df_label = df_label
            df_train = df_train.iloc[:, [0, 1, 2, 4]]
            df_dev = df_dev.iloc[:, [0, 1, 2, 4]]
            df_test = df_test.iloc[:, [0, 1, 2, 4]]
            self.df_data_labeled = pd.concat(
                [df_train, df_dev, df_test], ignore_index=True
            )
            print("---Loading Raw Json Files Finished!---")
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
                if attribute == "profile":
                    df_profile = pd.json_normalize(self.df_data_labeled["profile"][0])
                    for sub_attribute in df_profile.columns:
                        if sub_attribute == "id" or sub_attribute == "id_str":
                            continue
                        else:
                            attributes.append(sub_attribute)
                else:
                    attributes.append(attribute)
            attributes_dict = {val: index for index, val in enumerate(attributes)}
            with open(attributes_2_index_path, "w") as file:
                json.dump(attributes_dict, file)
                print("attributes_2_index.json file created!")
        else:
            print("attributes_2_index.json file existed!")

        if not os.path.exists(index_2_attributes_path):
            attributes = []
            for attribute in self.df_data_labeled.columns:
                if attribute == "profile":
                    df_profile = pd.json_normalize(self.df_data_labeled["profile"][0])
                    for sub_attribute in df_profile.columns:
                        if sub_attribute == "id" or sub_attribute == "id_str":
                            continue
                        else:
                            attributes.append(sub_attribute)
                else:
                    attributes.append(attribute)
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
                for row in tqdm(self.df_data_labeled.itertuples())
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
            if i == 0:
                continue

            elif i == 1:
                if value is None:
                    graph_list.append(none_embedding)
                    continue
                encoded_text = tokenizer(
                    str(value), truncation=True, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    outputs = model(**encoded_text)
                    id_embedding_pool = outputs.last_hidden_state.mean(dim=1).to(device)
                    graph_list.append(id_embedding_pool[0])

            elif i == 2:
                for j, sub_value in value.items():
                    if j == "id" or j == "id_str":
                        continue
                    else:
                        if sub_value is None:
                            graph_list.append(none_embedding)
                            continue
                        encoded_text = tokenizer(
                            str(sub_value), truncation=True, return_tensors="pt"
                        ).to(device)
                        with torch.no_grad():
                            outputs = model(**encoded_text)
                            sub_profile_embedding_pool = outputs.last_hidden_state.mean(
                                dim=1
                            ).to(device)
                            graph_list.append(sub_profile_embedding_pool[0])

            elif i == 3:
                if value is None:
                    graph_list.append(none_embedding)
                    continue
                for j, each_tweet in enumerate(value):
                    encoded_text = tokenizer(
                        str(each_tweet), truncation=True, return_tensors="pt"
                    ).to(device)
                    with torch.no_grad():
                        outputs = model(**encoded_text)
                        each_tweet_embedding_pool = outputs.last_hidden_state.mean(
                            dim=1
                        ).to(device)
                    if j == 0:
                        each_person_tweets_embedding_sum = each_tweet_embedding_pool[0]
                    else:
                        each_person_tweets_embedding_sum += each_tweet_embedding_pool[0]
                each_person_tweets_embedding_pool = (
                    each_person_tweets_embedding_sum / len(value)
                )
                graph_list.append(each_person_tweets_embedding_pool)

            elif i == 4:
                if value is None:
                    graph_list.append(none_embedding)
                    continue
                for j, each_domain in enumerate(value):
                    encoded_text = tokenizer(
                        str(each_domain), truncation=True, return_tensors="pt"
                    ).to(device)
                    with torch.no_grad():
                        outputs = model(**encoded_text)
                        each_domain_embedding_pool = outputs.last_hidden_state.mean(
                            dim=1
                        ).to(device)
                    if j == 0:
                        each_person_domains_embedding_sum = each_domain_embedding_pool[
                            0
                        ]
                    else:
                        each_person_domains_embedding_sum += each_domain_embedding_pool[
                            0
                        ]
                each_person_domains_embedding_pool = (
                    each_person_domains_embedding_sum / len(value)
                )
                graph_list.append(each_person_domains_embedding_pool)
                assert len(graph_list) == 39
        graph_tensor = torch.stack(graph_list).to(device)
        return graph_tensor

    def single_graph_edge_index(self, attributes_2_index):

        single_link = [
            ["ID", "name"],
            ["name", "followers_count"],
            ["name", "listed_count"],
            ["ID", "location"],
            ["name", "lang"],
            ["name", "profile_background_color"],
            ["profile_background_color", "profile_image_url"],
            ["profile_background_color", "profile_link_color"],
            ["ID", "profile_use_background_image"],
            ["name", "tweet"],
            ["name", "domain"],
            ["ID", "url"],
        ]

        complete_link = [
            ["name", "screen_name", "profile_location", "description"],
            ["followers_count", "friends_count"],
            ["listed_count", "favourites_count"],
            ["location", "created_at", "utc_offset", "time_zone", "geo_enabled"],
            [
                "statuses_count",
                "lang",
                "contributors_enabled",
                "is_translator",
                "is_translation_enabled",
            ],
            [
                "profile_background_color",
                "profile_background_image_url",
                "profile_background_image_url_https",
                "profile_background_tile",
            ],
            ["profile_image_url", "profile_image_url_https"],
            [
                "profile_link_color",
                "profile_sidebar_border_color",
                "profile_sidebar_fill_color",
                "profile_text_color",
            ],
            [
                "profile_use_background_image",
                "has_extended_profile",
                "default_profile",
                "default_profile_image",
            ],
            ["url", "entities", "protected", "verified"],
        ]
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
        dataset = "twibot-20"
        for depth in range(3, 9):
            tree.load_attribute_coding_tree(dataset, depth)
        print("---Attribute Graph Generated!---")

    def train_val_test_mask(self):
        train_idx = range(8278)
        val_idx = range(8278, 8278 + 2365)
        test_idx = range(8278 + 2365, 8278 + 2365 + 1183)
        return train_idx, val_idx, test_idx

    def dataloader(self):
        edge_index, edge_type = self.Build_Graph()
        train_idx, val_idx, test_idx = self.train_val_test_mask()
        return edge_index, edge_type, train_idx, val_idx, test_idx


if __name__ == "__main__":
    dataset = Twibot20_attribute_tree(device=device, process=True, save=True)
    dataset.attribute_graph_generate()
    dataset.attribute_tree_generate()
