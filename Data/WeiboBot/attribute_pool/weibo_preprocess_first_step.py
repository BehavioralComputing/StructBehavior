import pandas as pd
from pandas import json_normalize
import json
from misc.ToolFunc import two_significant_digits_strict, analyse_df_feature
import os


class PreprocessFirstStep:
    def __init__(self, root, device="cuda:0"):
        self.root = root
        self.device = device
        self.debug = True

        print("Loading labeled.json")
        df_labeled = pd.read_json(os.path.join(root, "weibo_labeled.json"))
        print("Loading support.json")
        df_support = pd.read_json(os.path.join(root, "weibo_support.json"))

        self.ori_data = pd.concat([df_labeled, df_support], ignore_index=True)

        self.ori_data.drop(
            columns=["id", "profile_url", "avatar_url", "label"], inplace=True
        )

    def __bin_num_properties(self):
        num_prop_cols = [
            "follows",
            "followers",
            "avg_like",
            "avg_comment",
            "avg_repost",
            "avg_image_count",
            "level",
        ]

        for col in num_prop_cols:
            print(f"Now binning {col}...")
            self.ori_data[col] = self.ori_data[col].apply(
                lambda x: two_significant_digits_strict(x, debug_msg=False)
            )

    def __binning(self):
        print("Now binning numerical properties...")
        self.__bin_num_properties()
        print("Numerical properties binning finished")

        print("Binning finished")

    def preprocess(self):
        self.__binning()

        return self.ori_data

    def save(self, path):
        analyse_df_feature(self.ori_data)
        self.ori_data.to_csv(path, index=False)
        print(f"Saved preprocessed data to {path}")


if __name__ == "__main__":
    preprocess = PreprocessFirstStep(root="../preprocessed_data/")

    preprocessed_data = preprocess.preprocess()

    preprocess.save("./weibo_preprocessed_data_first_step.csv")
