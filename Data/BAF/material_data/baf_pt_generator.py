import torch
import numpy as np
import pandas as pd
import os
import dgl
from baf_max1SE_new import knn_maxE1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class BAFPtGenerator:
    def __init__(self, data_dir: str, output_dir: str, device: str = "cuda:0"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print("Loading labeled.csv...")
        self.labeled_df = pd.read_csv(os.path.join(data_dir, "baf_labeled.csv"))
        print("Loading support.csv...")
        self.support_df = pd.read_csv(os.path.join(data_dir, "baf_support.csv"))
        print("Data loaded")

        self.data_df = pd.concat([self.labeled_df, self.support_df], ignore_index=True)

        print(f"Data shape: {self.data_df.shape}")

    def label_preprocess(self):
        print("Processing labels...")
        labels = self.labeled_df["fraud_bool"].to_numpy()
        labels = torch.tensor(labels, dtype=torch.int64).to(self.device)
        labels_path = os.path.join(self.output_dir, "baf_label.pt")
        torch.save(labels, labels_path)
        print(f"Labels saved to {labels_path}")

    def cat_prop_preprocess(self):
        print("Processing categorical properties...")

        bool_cats = [
            "email_is_free",
            "phone_home_valid",
            "phone_mobile_valid",
            "has_other_cards",
            "foreign_request",
            "keep_alive_session",
        ]

        bool_data = self.data_df[bool_cats].to_numpy()
        bool_data = torch.tensor(bool_data, dtype=torch.int64).to(self.device)

        other_cats = [
            "payment_type",
            "employment_status",
            "housing_status",
            "source",
            "device_os",
        ]

        other_cats_data = pd.get_dummies(self.data_df[other_cats], dtype=int).to_numpy()
        other_cats_data = torch.tensor(other_cats_data, dtype=torch.int64).to(
            self.device
        )

        cat_data = torch.cat((bool_data, other_cats_data), dim=1)
        cat_path = os.path.join(self.output_dir, "baf_cat_properties_tensor.pt")
        torch.save(cat_data, cat_path)
        print(f"Categorical properties saved to {cat_path}")

    def num_prop_preprocess(self):
        print("Processing numerical properties...")
        num_props = [
            "income",
            "name_email_similarity",
            "prev_address_months_count",
            "current_address_months_count",
            "customer_age",
            "days_since_request",
            "intended_balcon_amount",
            "zip_count_4w",
            "velocity_6h",
            "velocity_24h",
            "velocity_4w",
            "bank_branch_count_8w",
            "date_of_birth_distinct_emails_4w",
            "credit_risk_score",
            "bank_months_count",
            "proposed_credit_limit",
            "session_length_in_minutes",
            "device_distinct_emails_8w",
            "device_fraud_count",
            "month",
        ]

        num_data = self.data_df[num_props].to_numpy()
        num_data = torch.tensor(num_data, dtype=torch.float32).to(self.device)

        num_data[num_data < 0] = 0

        mean = num_data.mean(dim=0)
        std = num_data.std(dim=0)
        std[std == 0] = 1.0

        num_data = (num_data - mean) / std

        num_path = os.path.join(self.output_dir, "baf_num_properties_tensor.pt")
        torch.save(num_data, num_path)
        print(f"Numerical properties saved to {num_path}")

    def edge_preprocess(self):
        print("Loading Node_embeddings...")
        cat_tensor = torch.load(os.path.join(BASE_DIR, "baf_cat_properties_tensor.pt"))
        num_tensor = torch.load(os.path.join(BASE_DIR, "baf_num_properties_tensor.pt"))
        self.node_embed = torch.cat((cat_tensor, num_tensor), dim=1).to(self.device)

        print(self.node_embed.shape)

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

        edge_index_path = os.path.join(self.output_dir, "baf_edge_index.pt")
        torch.save(edge_index, edge_index_path)
        print(f"Edge index saved to {edge_index_path}")

        edge_type = torch.zeros((edge_index.shape[1],), dtype=torch.int64).to(
            self.device
        )
        edge_type_path = os.path.join(self.output_dir, "baf_edge_type.pt")
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
    generator = BAFPtGenerator(data_dir, output_dir)
    generator.run()
