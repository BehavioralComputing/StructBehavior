import pandas as pd
import os
import json


class PreprocessInstafake:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.labeled_in = f"{self.input_dir}/automatedAccountData.json"
        self.support_in = f"{self.input_dir}/nonautomatedAccountData.json"
        self.out = f"{self.output_dir}/instafake_preprocessed.json"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.load_data()

    def load_data(self):
        labeled_df = pd.read_json(self.labeled_in)
        support_df = pd.read_json(self.support_in)

        COLUMNS_TO_DROP = [
            "mediaCommentsAreDisabled",
            "mediaUploadTimes",
            "mediaHasLocationInfo",
        ]

        self.data_df = pd.concat([labeled_df, support_df], ignore_index=True)
        self.data_df = self.data_df.drop(columns=COLUMNS_TO_DROP)

        self.data_df["userHasHighlightReels"] = self.data_df[
            "userHasHighlighReels"
        ].fillna(False)
        self.data_df = self.data_df.drop(columns=["userHasHighlighReels"])

    def preprocess(self):
        assert (
            len(self.data_df["mediaLikeNumbers"])
            == len(self.data_df["mediaCommentNumbers"])
            == len(self.data_df["mediaHashtagNumbers"])
        )

        self.data_df["avgMediaLikeNum"] = self.data_df["mediaLikeNumbers"].apply(
            lambda x: sum(x) / len(x) if len(x) > 0 else 0
        )
        self.data_df["avgMediaCommentNum"] = self.data_df["mediaCommentNumbers"].apply(
            lambda x: sum(x) / len(x) if len(x) > 0 else 0
        )
        self.data_df["avgMediaHashtagNum"] = self.data_df["mediaHashtagNumbers"].apply(
            lambda x: sum(x) / len(x) if len(x) > 0 else 0
        )

        self.data_df["likeToCommentRatio"] = self.data_df[
            "avgMediaLikeNum"
        ] / self.data_df["avgMediaCommentNum"].replace(0, 1)

        self.data_df["followerToFollowingRatio"] = self.data_df[
            "userFollowerCount"
        ] / self.data_df["userFollowingCount"].replace(0, 1)

        COLUMNS_TO_DROP = [
            "mediaLikeNumbers",
            "mediaCommentNumbers",
            "mediaHashtagNumbers",
        ]

        self.data_df = self.data_df.drop(columns=COLUMNS_TO_DROP)

        label_temp = self.data_df.pop("automatedBehaviour")
        self.data_df["automatedBehaviour"] = label_temp

        with open(self.out, "w") as f:
            json.dump(self.data_df.to_dict(orient="records"), f, indent=4)


if __name__ == "__main__":
    input_dir = "../raw"
    output_dir = "."

    preprocessor = PreprocessInstafake(input_dir, output_dir)
    preprocessor.preprocess()
    print("Preprocessing completed successfully.")
