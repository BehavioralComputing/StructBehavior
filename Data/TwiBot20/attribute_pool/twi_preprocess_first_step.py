import pandas as pd
from pandas import json_normalize
import json
from misc.ToolFunc import (
    two_significant_digits,
    decompose_date,
    decompose_color,
    str_list_to_bool_dict,
    analyse_df_feature,
)
import os


class PreprocessFirstStep:
    def __init__(self, root, device="cuda:0"):
        self.root = root
        self.device = device
        self.debug = {
            "num_prop": False,
            "date_prop": False,
            "color_prop": False,
            "domain_prop": False,
        }

        print("Loading train.json")
        df_train = pd.read_json(os.path.join(root, "train.json"))
        print("Loading dev.json")
        df_dev = pd.read_json(os.path.join(root, "dev.json"))
        print("Loading test.json")
        df_test = pd.read_json(os.path.join(root + "test.json"))
        print("Loading support.json")
        df_support = pd.read_json(os.path.join(root, "support.json"))
        print("Finished")

        self.ori_data = pd.concat(
            [df_train, df_dev, df_test, df_support], ignore_index=True
        )

        self.__ori_to_flattened()

    def __ori_to_flattened(self):
        self.flatten_data = json_normalize(self.ori_data["profile"])

        self.flatten_data.drop(["id", "id_str"], axis=1, inplace=True)

        self.flatten_data["tweet"] = self.ori_data["tweet"]
        self.flatten_data["domain"] = self.ori_data["domain"]

        print("Flattened data shape:", self.flatten_data.shape)

        print("Flattened data columns:")
        for col in self.flatten_data.columns:
            print(col)

    def __bin_num_properties(self):
        num_prop_cols = [
            "followers_count",
            "friends_count",
            "listed_count",
            "favourites_count",
            "statuses_count",
        ]

        for col in num_prop_cols:
            print(f"Now binning {col}...")

            self.flatten_data[col] = self.flatten_data[col].apply(
                lambda x: two_significant_digits(x, debug_msg=self.debug["num_prop"])
            )

    def __bin_date_properties(self):
        date_prop_cols = ["created_at"]

        for col in date_prop_cols:
            print(f"Now binning {col}...")

            date_components = self.flatten_data[col].apply(
                lambda x: decompose_date(x, debug_msg=self.debug["date_prop"])
            )

            for key in date_components[0].keys():
                self.flatten_data[f"{col}_{key}"] = date_components.apply(
                    lambda x: x[key]
                )

            self.flatten_data.drop(col, axis=1, inplace=True)

    def __bin_color_properties(self):
        color_prop_cols = [
            "profile_background_color",
            "profile_link_color",
            "profile_sidebar_border_color",
            "profile_sidebar_fill_color",
            "profile_text_color",
        ]

        for col in color_prop_cols:
            print(f"Now binning {col}...")

            color_components = self.flatten_data[col].apply(
                lambda x: decompose_color(x, debug_msg=self.debug["color_prop"])
            )

            for key in color_components[0].keys():
                self.flatten_data[f"{col}_{key}"] = color_components.apply(
                    lambda x: x[key]
                )

            self.flatten_data.drop(col, axis=1, inplace=True)

    def __bin_domain_properties(self):
        domain_prop_cols = ["domain"]

        for col in domain_prop_cols:
            print(f"Now binning {col}...")

            domain_components = self.flatten_data[col].apply(
                lambda x: str_list_to_bool_dict(
                    x,
                    ["Politics", "Business", "Entertainment", "Sports"],
                    debug_msg=self.debug["domain_prop"],
                )
            )

            for key in domain_components[0].keys():
                self.flatten_data[f"{col}_{key}"] = domain_components.apply(
                    lambda x: x[key]
                )

            self.flatten_data.drop(col, axis=1, inplace=True)

    def __binning(self):
        print("Now binning numerical properties...")
        self.__bin_num_properties()
        print("Numerical properties binning finished")

        print("Now binning date properties...")
        self.__bin_date_properties()
        print("Date properties binning finished")

        print("Now binning color properties...")
        self.__bin_color_properties()
        print("Color properties binning finished")

        print("Now binning domain properties...")
        self.__bin_domain_properties()
        print("Domain properties binning finished")

        print("Binning finished")

    def preprocess(self):
        self.__binning()

        return self.flatten_data

    def save(self, path):
        analyse_df_feature(self.flatten_data)
        self.flatten_data.to_csv(path, index=False)
        print(f"Saved preprocessed data to {path}")


if __name__ == "__main__":
    preprocess = PreprocessFirstStep(root="../raw/")

    preprocessed_data = preprocess.preprocess()

    preprocess.save("./preprocessed_data_first_step.csv")
