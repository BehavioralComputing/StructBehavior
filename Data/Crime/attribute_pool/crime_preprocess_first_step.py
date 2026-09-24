import pandas as pd
from misc.ToolFunc import two_significant_digits_strict, analyse_df_feature
import torch
import os


class PreprocessFirstStep:
    def __init__(self, root="../preprocessed_data", device="cuda:0"):
        self.root = root
        self.device = device
        self.debug = False

        print("Loading labeled data...")
        self.labeled_data = pd.read_csv(os.path.join(self.root, "crime_labeled.csv"))
        print("Loading support data...")
        self.support_data = pd.read_csv(os.path.join(self.root, "crime_support.csv"))

        self.df_data = pd.concat(
            [self.labeled_data, self.support_data], ignore_index=True
        )

        self.df_data = self.df_data.drop(columns=["Label"])

        print("Base data loaded")

        print(f"Shape of data: {self.df_data.shape}")

    def __bin_num_properties(self):
        num_prop_cols = ["Date Difference"]

        for col in num_prop_cols:
            print(f"Now binning {col}...")
            self.df_data[col] = self.df_data[col].apply(
                lambda x: two_significant_digits_strict(x, debug_msg=self.debug)
            )

    def __binning(self):
        print("Now binning numerical properties...")
        self.__bin_num_properties()
        print("Numerical properties binning finished")

        print("Binning finished")

    def preprocess(self):
        self.__binning()

        if "fraud_bool" in self.df_data.columns:
            self.df_data = self.df_data.drop(columns=["fraud_bool"])
            print("Dropped 'fraud_bool' column")

        return self.df_data

    def save(self, path):
        analyse_df_feature(self.df_data)
        self.df_data.to_csv(path, index=False)
        print(f"Saved preprocessed data to {path}")


if __name__ == "__main__":
    preprocess = PreprocessFirstStep()

    preprocessed_data = preprocess.preprocess()

    preprocess.save("./crime_preprocessed_data_first_step.csv")
