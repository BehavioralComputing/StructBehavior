import torch
import numpy as np
import pandas as pd
import os
import dgl
from weibo_max1SE import knn_maxE1
from pandas import json_normalize
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class WEIBOPtGenerator:
    def __init__(self, data_dir: str, output_dir: str, device: str = "cuda:0"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print("Loading weibo_labeled.json...")
        self.labeled_df = pd.read_json(
            os.path.join(self.data_dir, "weibo_labeled.json")
        )
        print("Loading weibo_support.json...")
        self.support_df = pd.read_json(
            os.path.join(self.data_dir, "weibo_support.json")
        )
        self.base_df = pd.concat([self.labeled_df, self.support_df], ignore_index=True)
        print("Data loaded")

        print(f"Data shape: {self.base_df.shape}")

    def post_preprocess(self):
        if os.path.exists(os.path.join(self.output_dir, "weibo_posts_tensor.pt")):
            print("Posts tensor already exists. Skipping preprocessing.")
            return

        print("Processing posts...")

        model_name = f"{os.path.join(BASE_DIR, '../../../bert/bert-base-chinese')}"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        model.to(self.device)

        posts_list = []

        post_column = self.base_df["posts"]

        for posts in tqdm(post_column):
            posts = json_normalize(posts)

            for i, cleaned_content in enumerate(posts["cleaned_content"]):
                encoded_text = tokenizer(
                    cleaned_content,
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

                if i == 0:
                    each_user_tensor = semantic_vector
                else:
                    each_user_tensor += semantic_vector

            each_user_tensor /= len(posts)
            posts_list.append(each_user_tensor)

        posts_tensor = torch.stack(posts_list, dim=0).to(self.device)
        posts_tensor = posts_tensor.squeeze(1)
        print(f"Posts tensor shape: {posts_tensor.shape}")
        posts_path = os.path.join(self.output_dir, "weibo_posts_tensor.pt")
        torch.save(posts_tensor, posts_path)
        print(f"Posts tensor saved to {posts_path}")

    def label_preprocess(self):
        if os.path.exists(os.path.join(self.output_dir, "weibo_label.pt")):
            print("Labels tensor already exists. Skipping preprocessing.")
            return

        print("Processing labels...")
        labels = self.labeled_df["label"].to_numpy()
        labels = torch.tensor(labels, dtype=torch.int64).to(self.device)
        labels_path = os.path.join(self.output_dir, "weibo_label.pt")
        torch.save(labels, labels_path)
        print(f"Labels saved to {labels_path}")

    def cat_prop_preprocess(self):
        if os.path.exists(
            os.path.join(self.output_dir, "weibo_cat_properties_tensor.pt")
        ):
            print(
                "Categorical properties tensor already exists. Skipping preprocessing."
            )
            return

        print("Processing categorical properties...")

        bool_cats = ["verified", "gender"]

        self.base_df["gender"] = self.base_df["gender"].apply(
            lambda x: 1 if x == "m" else 0
        )

        self.base_df["verified"] = self.base_df["verified"].apply(
            lambda x: 1 if x is True else 0
        )

        bool_data = self.base_df[bool_cats].to_numpy()
        bool_data = torch.tensor(bool_data, dtype=torch.int64).to(self.device)

        cat_data = bool_data
        cat_path = os.path.join(self.output_dir, "weibo_cat_properties_tensor.pt")
        torch.save(cat_data, cat_path)
        print(f"Categorical properties saved to {cat_path}")

    def num_prop_preprocess(self):
        if os.path.exists(
            os.path.join(self.output_dir, "weibo_num_properties_tensor.pt")
        ):
            print("Numerical properties tensor already exists. Skipping preprocessing.")
            return

        print("Processing numerical properties...")
        num_props = [
            "follows",
            "followers",
            "level",
            "avg_like",
            "avg_comment",
            "avg_repost",
            "avg_image_count",
        ]

        num_data = self.base_df[num_props].to_numpy()
        num_data = torch.tensor(num_data, dtype=torch.float32).to(self.device)

        print(f"Numerical data shape: {num_data.shape}")

        mean = num_data.mean(dim=0)
        std = num_data.std(dim=0)
        print(f"Mean: {mean}, Std: {std}")
        std[std == 0] = 1.0

        num_data = (num_data - mean) / std
        print(
            f"num_data shape: {num_data.shape}, mean: {num_data.mean(dim=0)}, std: {num_data.std(dim=0)}"
        )
        num_path = os.path.join(self.output_dir, "weibo_num_properties_tensor.pt")
        torch.save(num_data, num_path)
        print(f"Numerical properties saved to {num_path}")

    def edge_preprocess(self):
        if os.path.exists(os.path.join(self.output_dir, "weibo_edge_index.pt")):
            print("Edge index already exists. Skipping preprocessing.")
            return

        print("Loading Node_embeddings...")
        cat_tensor = torch.load(
            os.path.join(BASE_DIR, "weibo_cat_properties_tensor.pt")
        )
        num_tensor = torch.load(
            os.path.join(BASE_DIR, "weibo_num_properties_tensor.pt")
        )
        posts_tensor = torch.load(os.path.join(BASE_DIR, "weibo_posts_tensor.pt"))
        self.node_embed = torch.cat((cat_tensor, num_tensor, posts_tensor), dim=1).to(
            self.device
        )

        print("Finding the best k for k-NN graph...")

        best_k = knn_maxE1(self.node_embed, self.device)

        print("Processing edges...")

        print(f"Using k={best_k} for k-NN graph construction")

        if self.device == torch.device("cpu"):
            knn_g = dgl.knn_graph(
                self.node_embed, best_k, algorithm="bruteforce", dist="cosine"
            )
        else:
            knn_g = dgl.knn_graph(
                self.node_embed, best_k, algorithm="bruteforce-sharemem", dist="cosine"
            )

        knn_g = dgl.add_reverse_edges(knn_g)

        src, dst = knn_g.edges()

        edge_index = torch.stack((src, dst), dim=0)

        edge_index = torch.unique(edge_index, dim=1)
        print(f"Edge index shape: {edge_index.shape}")

        edge_index_path = os.path.join(self.output_dir, "weibo_edge_index.pt")
        torch.save(edge_index, edge_index_path)
        print(f"Edge index saved to {edge_index_path}")

        edge_type = torch.zeros((edge_index.shape[1],), dtype=torch.int64).to(
            self.device
        )
        edge_type_path = os.path.join(self.output_dir, "weibo_edge_type.pt")
        torch.save(edge_type, edge_type_path)
        print(f"Edge type saved to {edge_type_path}")

    def run(self):
        self.label_preprocess()
        self.cat_prop_preprocess()
        self.num_prop_preprocess()
        self.post_preprocess()
        self.edge_preprocess()
        print("All preprocessing steps completed.")


if __name__ == "__main__":
    data_dir = "../raw/"
    output_dir = "."
    generator = WEIBOPtGenerator(data_dir, output_dir)
    generator.run()
