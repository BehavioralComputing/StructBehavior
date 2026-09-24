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


class InstaBehavior:
    def __init__(self, root="./", device="cuda:0", mid_generated=False):
        self.root = root
        self.device = device
        self.mid_generated = mid_generated

        if not mid_generated:
            self.data_df = pd.read_csv(
                os.path.join(self.root, "insta_preprocessed_data_second_step.csv"),
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
            ["usernameLength", "userBiographyLength"],
            ["avgMediaLikeNum", "userFollowerCount"],
            ["userHasHighlightReels", "likeToCommentRatio"],
            ["userMediaCount", "avgMediaLikeNum"],
        ]

        complete_link = [
            ["userMediaCount", "userTagsCount"],
            [
                "avgMediaLikeNum",
                "avgMediaCommentNum",
                "avgMediaHashtagNum",
                "likeToCommentRatio",
            ],
            ["userFollowerCount", "userFollowingCount", "followerToFollowingRatio"],
            ["userHasHighlightReels", "userHasExternalUrl", "userBiographyLength"],
            ["usernameLength", "usernameDigitCount"],
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
            with open(os.path.join(BASE_DIR, "./insta_column_encoding.json"), "r") as f:
                self.column_encoding = json.load(f)
            print("insta_column_encoding.json loaded")

            self.data_df = pd.read_csv(os.path.join(BASE_DIR, "./insta_data_df.csv"))

            print("insta_data_df.csv loaded")

    def __save_mid_process(self):
        with open(os.path.join(BASE_DIR, "./insta_column_encoding.json"), "w") as f:
            json.dump(self.column_encoding, f)
        print("insta_column_encoding.json saved")

        self.data_df.to_csv(os.path.join(BASE_DIR, "./insta_data_df.csv"), index=False)
        print("insta_data_df.csv saved")

    def __connect_graph(self):
        data_dict = {}

        for graph in self.graph_all:
            complete_graph_edge(data_dict, graph, self.data_df)

        self.hetero_graph = dgl.heterograph(data_dict).to(self.device)

        print("Graph connected")

    def __decorate_text(self, col: str, val: str):
        if col == "userMediaCount":
            return f"User has {val} media posts."
        elif col == "userFollowerCount":
            return f"User has {val} followers."
        elif col == "userFollowingCount":
            return f"User is following {val} accounts."
        elif col == "userHasExternalUrl":
            if val == "1":
                return "User has an external URL in their profile."
            elif val == "0":
                return "User does not have an external URL in their profile."
            else:
                raise ValueError(f"Unexpected value for userHasExternalUrl: {val}")
        elif col == "userTagsCount":
            return f"User has {val} photos tagged by others."
        elif col == "userBiographyLength":
            return f"User's biography length is {val} characters."
        elif col == "usernameLength":
            return f"User's username length is {val} characters."
        elif col == "usernameDigitCount":
            return f"User's username contains {val} digits."
        elif col == "userHasHighlightReels":
            if val == "1":
                return "User has highlight reels."
            elif val == "0":
                return "User does not have highlight reels."
            else:
                raise ValueError(f"Unexpected value for userHasHighlightReels: {val}")
        elif col == "avgMediaLikeNum":
            return f"Average number of likes for user's media posts is {val}."
        elif col == "avgMediaCommentNum":
            return f"Average number of comments for user's media posts is {val}."
        elif col == "avgMediaHashtagNum":
            return f"Average number of hashtags for user's media posts is {val}."
        elif col == "likeToCommentRatio":
            return f"User's media posts have a like-to-comment ratio of {val}."
        elif col == "followerToFollowingRatio":
            return f"User's follower-to-following ratio is {val}."
        else:
            raise ValueError(f"Unexpected column name: {col}")

    def __gen_embedding(self):
        for col in self.column_encoding.keys():
            print("Now processing feature: ", col)
            feature_list = self.column_encoding[col].keys()
            feature_list_strings = list(map(str, feature_list))

            model_name = f"{os.path.join(BASE_DIR, '../../../bert/bert-base-uncased')}"
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name).to(self.device)
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
            os.path.join(BASE_DIR, "./insta_Data_hetero_graph.dgl"), [self.hetero_graph]
        )
        print(self.hetero_graph)


if __name__ == "__main__":
    twi = InstaBehavior(mid_generated=False)
