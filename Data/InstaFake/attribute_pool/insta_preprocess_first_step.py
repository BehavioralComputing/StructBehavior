import pandas as pd
from pandas import json_normalize
import json
from misc.ToolFunc import two_significant_digits_strict, analyse_df_feature


class PreprocessFirstStep:
    def __init__(self, root="../preprocessed_data/", device="cuda:0"):
        self.root = root
        self.device = device
        self.debug = True

        self.ori_data = pd.read_json(root + "instafake_preprocessed.json")

        self.ori_data.drop(columns=["automatedBehaviour"], inplace=True)

    def __bin_num_properties(self):
        num_prop_cols = [
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
    preprocess = PreprocessFirstStep()

    preprocessed_data = preprocess.preprocess()

    preprocess.save("./insta_preprocessed_data_first_step.csv")
