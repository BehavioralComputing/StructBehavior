import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import dgl
from crime_max1SE_new import knn_maxE1
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CRIMEPtGenerator:
    def __init__(self, data_dir: str, output_dir: str, device: str = "cuda:0"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print("Loading crime_labeled.csv...")
        self.labeled_df = pd.read_csv(os.path.join(self.data_dir, "crime_labeled.csv"))
        print("Loading crime_support.csv...")
        self.support_df = pd.read_csv(os.path.join(self.data_dir, "crime_support.csv"))
        self.base_df = pd.concat([self.labeled_df, self.support_df], ignore_index=True)
        print("Data loaded")

        print(f"Data shape: {self.base_df.shape}")
        print(self.base_df.info())

        seed_everything(42)

    def label_preprocess(self):
        if os.path.exists(os.path.join(self.output_dir, "crime_label.pt")):
            print("Labels already processed. Skipping label preprocessing.")
            return

        print("Processing labels...")
        labels = self.labeled_df["Label"].to_numpy()
        labels = torch.tensor(labels, dtype=torch.int64).to(self.device)
        labels_path = os.path.join(self.output_dir, "crime_label.pt")
        torch.save(labels, labels_path)
        print(f"Labels shape: {labels.shape}")
        print(f"Labels saved to {labels_path}")

    def __get_embedding_dim(self, num_categories: int) -> int:
        return min(50, (int(num_categories**0.25 * 4)))

    def cat_prop_preprocess(self):
        if os.path.exists(
            os.path.join(self.output_dir, "crime_cat_properties_tensor.pt")
        ):
            print(
                "Categorical properties already processed. Skipping categorical preprocessing."
            )
            return

        print("Processing categorical properties...")

        other_cats = [
            "AREA NAME",
            "Rpt Dist No",
            "Part 1-2",
            "Vict Sex",
            "Vict Descent",
            "Premis Desc",
            "Weapon Desc",
            "Status",
            "Month_Rptd",
            "Day_Rptd",
            "Year_Rptd",
            "Month_OCC",
            "Day_OCC",
            "Year_OCC",
            "Hour",
            "Minute",
        ]

        cat_fields = {cat: self.base_df[cat].nunique() for cat in other_cats}
        print(f"Categorical fields: {cat_fields}")

        embedding_dims = {
            cat: self.__get_embedding_dim(num) for cat, num in cat_fields.items()
        }
        print(f"Embedding dimensions: {embedding_dims}")

        embeddings = nn.ModuleDict(
            {
                cat: nn.Embedding(num_classes, embedding_dims[cat])
                for cat, num_classes in cat_fields.items()
            }
        )

        cat_data = []
        for cat in other_cats:
            cat_tensor = torch.tensor(
                self.base_df[cat].astype("category").cat.codes.values, dtype=torch.int64
            )

            cat_tensor = embeddings[cat](cat_tensor).to(self.device)

            print(
                f"{cat} tensor shape: {cat_tensor.shape}, embedding dim: {embedding_dims[cat]}"
            )

            cat_data.append(cat_tensor)

        cat_embedded = torch.cat(cat_data, dim=1)

        print(f"Combined categorical properties shape: {cat_embedded.shape}")

        cat_path = os.path.join(self.output_dir, "crime_cat_properties_tensor.pt")
        torch.save(cat_embedded, cat_path)
        print(f"Categorical properties saved to {cat_path}")

    def num_prop_preprocess(self):
        if os.path.exists(
            os.path.join(self.output_dir, "crime_num_properties_tensor.pt")
        ):
            print(
                "Numerical properties already processed. Skipping numerical preprocessing."
            )
            return

        print("Processing numerical properties...")
        num_props = ["Vict Age", "Date Difference"]

        num_data = self.base_df[num_props].fillna(0.0).to_numpy()
        num_data = torch.tensor(num_data, dtype=torch.float32).to(self.device)

        print(f"Numerical data shape: {num_data.shape}")

        mean = num_data.mean(dim=0)
        std = num_data.std(dim=0)
        std[std == 0] = 1.0

        num_data = (num_data - mean) / std
        num_path = os.path.join(self.output_dir, "crime_num_properties_tensor.pt")
        torch.save(num_data, num_path)
        print(f"Numerical properties saved to {num_path}")

    def edge_preprocess(self):
        if os.path.exists(
            os.path.join(self.output_dir, "crime_edge_index.pt")
        ) and os.path.exists(os.path.join(self.output_dir, "crime_edge_type.pt")):
            print("Edges already processed. Skipping edge preprocessing.")
            return

        print("Loading Node_embeddings...")
        cat_tensor = torch.load(
            os.path.join(BASE_DIR, "crime_cat_properties_tensor.pt")
        )
        num_tensor = torch.load(
            os.path.join(BASE_DIR, "crime_num_properties_tensor.pt")
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

        edge_index_path = os.path.join(self.output_dir, "crime_edge_index.pt")
        torch.save(edge_index, edge_index_path)
        print(f"Edge index saved to {edge_index_path}")

        edge_type = torch.zeros((edge_index.shape[1],), dtype=torch.int64).to(
            self.device
        )
        edge_type_path = os.path.join(self.output_dir, "crime_edge_type.pt")
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
    output_dir = "."
    generator = CRIMEPtGenerator(data_dir, output_dir)
    generator.run()
