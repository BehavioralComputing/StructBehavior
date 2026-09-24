import torch
import numpy as np
import pandas as pd
import os
import dgl
from insta_max1SE import knn_maxE1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class INSTAPtGenerator:
    def __init__(self, data_dir: str, output_dir: str, device: str = "cuda:0"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print("Loading insta_labeled.json...")
        self.base_df = pd.read_json(
            os.path.join(self.data_dir, "instafake_preprocessed.json")
        )
        print("Data loaded")

        print(f"Data shape: {self.base_df.shape}")

    def label_preprocess(self):
        if os.path.exists(os.path.join(self.output_dir, "insta_label.pt")):
            print("Labels already processed. Skipping label preprocessing.")
            return

        print("Processing labels...")
        labels = self.base_df["automatedBehaviour"].to_numpy()
        labels = torch.tensor(labels, dtype=torch.int64).to(self.device)
        labels_path = os.path.join(self.output_dir, "insta_label.pt")
        torch.save(labels, labels_path)
        print(f"Labels saved to {labels_path}")

    def cat_prop_preprocess(self):
        if os.path.exists(
            os.path.join(self.output_dir, "insta_cat_properties_tensor.pt")
        ):
            print(
                "Categorical properties already processed. Skipping categorical property preprocessing."
            )
            return

        print("Processing categorical properties...")

        bool_cats = ["userHasHighlightReels", "userHasExternalUrl"]

        bool_data = self.base_df[bool_cats].to_numpy()
        bool_data = torch.tensor(bool_data, dtype=torch.int64).to(self.device)

        cat_data = bool_data
        cat_path = os.path.join(self.output_dir, "insta_cat_properties_tensor.pt")
        torch.save(cat_data, cat_path)
        print(f"Categorical properties saved to {cat_path}")

    def num_prop_preprocess(self):
        if os.path.exists(
            os.path.join(self.output_dir, "insta_num_properties_tensor.pt")
        ):
            print(
                "Numerical properties already processed. Skipping numerical property preprocessing."
            )
            return

        print("Processing numerical properties...")
        num_props = [
            "userMediaCount",
            "userFollowerCount",
            "userFollowingCount",
            "userTagsCount",
            "userBiographyLength",
            "usernameLength",
            "usernameDigitCount",
            "avgMediaLikeNum",
            "avgMediaCommentNum",
            "avgMediaHashtagNum",
            "likeToCommentRatio",
            "followerToFollowingRatio",
        ]

        num_data = self.base_df[num_props].to_numpy()
        num_data = torch.tensor(num_data, dtype=torch.float32).to(self.device)

        print(f"Numerical data shape: {num_data.shape}")

        mean = num_data.mean(dim=0)
        std = num_data.std(dim=0)
        std[std == 0] = 1.0

        num_data = (num_data - mean) / std
        num_path = os.path.join(self.output_dir, "insta_num_properties_tensor.pt")
        torch.save(num_data, num_path)
        print(f"Numerical properties saved to {num_path}")

    def edge_preprocess(self):
        if os.path.exists(os.path.join(self.output_dir, "insta_edge_index.pt")):
            print("Edges already processed. Skipping edge preprocessing.")
            return
        print("Loading Node_embeddings...")
        cat_tensor = torch.load(
            os.path.join(BASE_DIR, "insta_cat_properties_tensor.pt")
        )
        num_tensor = torch.load(
            os.path.join(BASE_DIR, "insta_num_properties_tensor.pt")
        )
        self.node_embed = torch.cat((cat_tensor, num_tensor), dim=1).to(self.device)

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

        edge_index_path = os.path.join(self.output_dir, "insta_edge_index.pt")
        torch.save(edge_index, edge_index_path)
        print(f"Edge index saved to {edge_index_path}")

        edge_type = torch.zeros((edge_index.shape[1],), dtype=torch.int64).to(
            self.device
        )
        edge_type_path = os.path.join(self.output_dir, "insta_edge_type.pt")
        torch.save(edge_type, edge_type_path)
        print(f"Edge type saved to {edge_type_path}")

    def run(self):
        self.label_preprocess()
        self.cat_prop_preprocess()
        self.num_prop_preprocess()
        self.edge_preprocess()
        print("All preprocessing steps completed.")


if __name__ == "__main__":
    data_dir = "../preprocessed_data/"
    output_dir = "./"
    generator = INSTAPtGenerator(data_dir, output_dir)
    generator.run()
