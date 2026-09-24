import pandas as pd
import dgl
import torch
import math
import numpy as np
from transformers import BertTokenizer, BertModel
import json
from datetime import datetime as dt
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def del_nan(A: list, B: list):
    array_A = np.array(A)
    array_B = np.array(B)
    nan_mask = np.isnan(array_A) | np.isnan(array_B)
    valid_indices = ~nan_mask
    return array_A[valid_indices], array_B[valid_indices]


def complete_graph_edge(
    data_dict: dict, columns_to_convert: list, data_df: pd.DataFrame
):
    for src in columns_to_convert:
        for tar in columns_to_convert:
            if src != tar:
                new_list_A, new_list_B = del_nan(data_df[src], data_df[tar])
                data_dict.update(
                    {
                        (src, "edge_" + src + "_" + tar, tar): (
                            torch.tensor(new_list_A, dtype=torch.int),
                            torch.tensor(new_list_B, dtype=torch.int),
                        )
                    }
                )


class TwiBehavior:
    def __init__(self, root="./", device="cuda:0", mid_generated=False):
        self.root = root
        self.device = device
        self.mid_generated = mid_generated

        if not mid_generated:
            self.data_df = pd.read_csv(
                os.path.join(self.root, "twi_preprocessed_data_second_step.csv"),
                encoding="utf-8",
            )
            self.data_df = self.data_df.astype("string")

        self.column_encoding = {}
        self.graph_all = []
        self.hetero_graph = None

        self.__get_graph_all()
        self.__attr_val_to_index()
        self.__connect_graph()
        self.__gen_embedding()

    def __get_graph_all(self):
        single_link = [
            ["name", "followers_count"],
            ["name", "listed_count"],
            ["name", "lang"],
            [
                "name",
                "profile_background_color_red",
                "profile_background_color_green",
                "profile_background_color_blue",
            ],
            [
                "profile_background_color_red",
                "profile_background_color_green",
                "profile_background_color_blue",
                "profile_image_url_https",
            ],
            [
                "profile_background_color_red",
                "profile_background_color_green",
                "profile_background_color_blue",
                "profile_link_color_red",
                "profile_link_color_green",
                "profile_link_color_blue",
            ],
            ["name", "tweet"],
            [
                "name",
                "domain_Politics",
                "domain_Business",
                "domain_Entertainment",
                "domain_Sports",
            ],
        ]

        complete_link = [
            ["name", "screen_name", "profile_location", "description"],
            ["followers_count", "friends_count"],
            ["listed_count", "favourites_count"],
            [
                "location",
                "created_at_year",
                "created_at_month",
                "created_at_day",
                "created_at_hour",
                "created_at_minute",
                "created_at_second",
                "utc_offset",
                "time_zone",
                "geo_enabled",
            ],
            [
                "statuses_count",
                "lang",
                "contributors_enabled",
                "is_translator",
                "is_translation_enabled",
            ],
            [
                "profile_background_color_red",
                "profile_background_color_green",
                "profile_background_color_blue",
                "profile_background_image_url_https",
                "profile_background_tile",
            ],
            [
                "profile_link_color_red",
                "profile_link_color_green",
                "profile_link_color_blue",
            ],
            [
                "profile_sidebar_border_color_red",
                "profile_sidebar_border_color_green",
                "profile_sidebar_border_color_blue",
            ],
            [
                "profile_sidebar_fill_color_red",
                "profile_sidebar_fill_color_green",
                "profile_sidebar_fill_color_blue",
            ],
            [
                "profile_text_color_red",
                "profile_text_color_green",
                "profile_text_color_blue",
            ],
            [
                "profile_use_background_image",
                "has_extended_profile",
                "default_profile",
                "default_profile_image",
            ],
            ["url", "entities", "protected", "verified"],
        ]

        self.graph_all = single_link + complete_link
        print("Graph all:", self.graph_all)

    def __attr_val_to_index(self):
        if not self.mid_generated:
            for p_col in self.data_df.columns:
                print("Handling: ", p_col)

                unique_values = self.data_df[p_col].unique()
                print("length of unique values: ", len(unique_values))

                encoding = {value: i for i, value in enumerate(unique_values)}

                self.column_encoding[p_col] = encoding

            for col, enc in self.column_encoding.items():
                self.data_df[col] = self.data_df[col].map(enc)

            self.__save_mid_process()
        else:
            with open(os.path.join(BASE_DIR, "./column_encoding.json"), "r") as f:
                self.column_encoding = json.load(f)
            print("column_encoding.json loaded")

            self.data_df = pd.read_csv(os.path.join(BASE_DIR, "./data_df.csv"))

            print("data_df.csv loaded")

    def __save_mid_process(self):
        with open(os.path.join(BASE_DIR, "./column_encoding.json"), "w") as f:
            json.dump(self.column_encoding, f)
        print("column_encoding.json saved")

        self.data_df.to_csv(os.path.join(BASE_DIR, "./data_df.csv"), index=False)
        print("data_df.csv saved")

    def __connect_graph(self):
        data_dict = {}

        for graph in self.graph_all:
            complete_graph_edge(data_dict, graph, self.data_df)

        self.hetero_graph = dgl.heterograph(data_dict).to(self.device)

        print("Graph connected")

    def __decorate_text(self, col: str, val: str):

        if "color" in col:
            if val == "Unknown":
                return f"The user's {col} component is unknown"
            else:
                return f"The user's {col} component(0-255): {int(float(val))}"
        elif "domain" in col:
            if val == "True":
                return f"The user is in {col.split('_')[1]} domain"
            else:
                return f"The user is not in {col.split('_')[1]} domain"
        elif "default" in col:
            if val == "True":
                return f"The user is using the {col}"
            elif val == "False":
                return f"The user is not using the {col}"
            else:
                return f"We don't know whether the user is using the {col}"
        elif "created" in col:
            if val == "Unknown":
                return f"The user's created {col.split('_')[-1]} is unknown"
            else:
                return f"The user's created {col.split('_')[-1]} is {int(float(val))}"
        elif "count" in col:
            if val == "Unknown":
                return f"The user's {col.split('_')[0]} count is unknown"
            else:
                return f"The user's {col.split('_')[0]} count is {int(float(val))}"
        elif "profile_background_image_url" in col:
            if val == "Unknown":
                return "The user's background image theme is unknown"
            elif val == "False":
                return "The user has not set a background image theme"
            else:
                return f"The user's background image theme is {val}"
        elif "is_translator" == col:
            if val == "True":
                return "The user is a translator for Twitter"
            elif val == "False":
                return "The user is not a translator for Twitter"
            else:
                return "We don't know whether the user is a translator for Twitter"
        elif "enabled" in col:
            if val == "True":
                return f"The user enabled {col.split('_')[-2]}"
            elif val == "False":
                return f"The user disabled {col.split('_')[-2]}"
            else:
                return f"We don't know whether the user enabled {col.split('_')[-2]}"
        elif "entities" == col:
            if val == "True":
                return "The user has links in their profile"
            elif val == "False":
                return "The user does not have links in their profile"
            else:
                return "We don't know whether the user has links in their profile"
        else:
            if val == "True":
                return f"The user has/enabled {col}"
            elif val == "False":
                return f"The user does not have/enabled {col}"
            elif val == "Unknown":
                return f"We don't know whether the user has/enabled {col}"
            else:
                raise ValueError(f"Unexpected value: {val} for column: {col}")

    def __gen_embedding(self):
        for col in self.column_encoding.keys():
            print("Now processing feature: ", col)
            feature_list = self.column_encoding[col].keys()
            feature_list_strings = list(map(str, feature_list))

            model_name = f"{os.path.join(BASE_DIR, '../../../bert/bert-base-uncased')}"
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
            model.to(self.device)
            semantic_vectors = []
            feature_len = len(feature_list_strings)
            counter = 0
            for value in feature_list_strings:

                counter += 1
                if counter % 100 == 0:
                    print(f"Processing {counter}/{feature_len} for {col}...")

                value = self.__decorate_text(col, value)

                encoded_text = tokenizer(
                    value,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                with torch.no_grad():
                    outputs = model(**encoded_text)
                    semantic_vector = outputs.last_hidden_state.mean(dim=1).to(
                        self.device
                    )
                    semantic_vectors.append(semantic_vector)
            self.hetero_graph.nodes[col].data["feat"] = torch.cat(
                semantic_vectors, dim=0
            ).to(self.device)

            print("Feature embedding for ", col, " generated")

        dgl.save_graphs(
            os.path.join(BASE_DIR, "./twi_Data_hetero_graph.dgl"), [self.hetero_graph]
        )
        print(self.hetero_graph)


if __name__ == "__main__":
    twi = TwiBehavior(mid_generated=False)
