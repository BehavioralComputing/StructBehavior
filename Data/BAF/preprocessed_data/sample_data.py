import pandas as pd
import os


class SampleData:
    def __init__(self, in_file, out_path):
        self.in_file = in_file
        self.out_path = out_path
        self.df = pd.read_csv(in_file)

    def sample_with_month(self, sample_ratio=0.1):
        grouped = self.df.groupby("month")

        sampled_df = grouped.apply(
            lambda x: x.sample(frac=sample_ratio, random_state=42)
        ).reset_index(drop=True)

        self.df = sampled_df

        print(f"Dataset shape after sampling: {self.df.shape}")

        print("fraud_bool stats:")
        print(self.df["fraud_bool"].value_counts())

        print("samples by month:")
        print(self.df["month"].value_counts())

    def sample_with_fixed_ratio(self, sample_ratio=0.1):
        labeled_df = self.df.sample(frac=sample_ratio, random_state=42)
        support_df = self.df.drop(labeled_df.index)

        print(f"labeled df: {labeled_df.shape} support df: {support_df.shape}")
        print(
            f"labeled_df['fraud_bool'].value_counts(): {labeled_df['fraud_bool'].value_counts()}"
        )
        print(
            f"support_df['fraud_bool'].value_counts(): {support_df['fraud_bool'].value_counts()}"
        )

        support_df = support_df.drop(columns=["fraud_bool"])

        labeled_df.to_csv(os.path.join(self.out_path, "baf_labeled.csv"), index=False)
        support_df.to_csv(os.path.join(self.out_path, "baf_support.csv"), index=False)


if __name__ == "__main__":
    in_file = "../raw/Base.csv"
    out_path = "."

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    sampler = SampleData(in_file, out_path)
    sampler.sample_with_month(sample_ratio=0.1)
    sampler.sample_with_fixed_ratio(sample_ratio=0.1)
