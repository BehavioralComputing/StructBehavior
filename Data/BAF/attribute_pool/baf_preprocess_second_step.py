import pandas as pd
from misc.ToolFunc import analyse_df_feature
import os


class PreprocessSecondStep:
    def __init__(self, root="./", device="cuda:0"):
        self.root = root
        self.device = device
        self.debug = True

        print(f"Loading data from {self.root}baf_preprocessed_data_first_step.csv")
        self.df_data = pd.read_csv(
            os.path.join(self.root, "baf_preprocessed_data_first_step.csv")
        )
        print("Finished loading data")

    def __reduce_too_many_unique_values(self):
        threshold = 1000

        count = 0
        for column in self.df_data.columns:
            print(f"\nNow processing {column}\n")
            unique_size = self.df_data[column].nunique()
            if unique_size > threshold:
                count += 1
                print(
                    f"Column {column} with type {self.df_data[column].dtype} has {unique_size} unique values, converting to bool"
                )
                self.df_data[column] = self.df_data[column].apply(
                    lambda x: "True" if x != "Unknown" and x != "False" else x
                )
                self.df_data[column] = self.df_data[column].astype("string")

        if count == 0:
            print("No columns with unique values greater than threshold")
        else:
            print(
                f"Converted {count} columns with unique values greater than {threshold} to bool type"
            )

    def __all_columns_to_str(self):
        num_cols = self.df_data.select_dtypes(include=["number"]).columns
        print(f"Converting {len(num_cols)} numerical columns to string")

        self.df_data[num_cols] = self.df_data[num_cols].map(
            lambda x: "Unknown" if x < 0 else x
        )

        for column in self.df_data.columns:
            self.df_data[column] = self.df_data[column].apply(lambda x: str(x).strip())
            self.df_data[column] = self.df_data[column].astype("string")

    def preprocess(self):
        self.__all_columns_to_str()
        self.__reduce_too_many_unique_values()

        analyse_df_feature(self.df_data, threshold=1000)
        print(f"Saving data to {self.root}baf_preprocessed_data_second_step.csv")
        self.df_data.to_csv(
            os.path.join(self.root, "baf_preprocessed_data_second_step.csv"),
            index=False,
        )


if __name__ == "__main__":
    preprocessor = PreprocessSecondStep()
    preprocessor.preprocess()
