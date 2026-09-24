import pandas as pd
from ToolFunc import analyse_df_feature
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class CrimeDataPreprocessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)

    def __find_top_n(self, n: int = 10, drop=True):
        top_10_crime_codes = self.df["Crm Cd"].value_counts().nlargest(n + 1).index

        if drop:
            top_10_crime_codes = top_10_crime_codes[1:]
        else:
            top_10_crime_codes = top_10_crime_codes[:n]

        self.df = self.df[self.df["Crm Cd"].isin(top_10_crime_codes)]

    def __preprocess_time(self):
        self.df["Date Rptd"] = pd.to_datetime(self.df["Date Rptd"])
        self.df["Month_Rptd"] = self.df["Date Rptd"].dt.month
        self.df["Day_Rptd"] = self.df["Date Rptd"].dt.day
        self.df["Year_Rptd"] = self.df["Date Rptd"].dt.year

        self.df["DATE OCC"] = pd.to_datetime(self.df["DATE OCC"])
        self.df["Month_OCC"] = self.df["DATE OCC"].dt.month
        self.df["Day_OCC"] = self.df["DATE OCC"].dt.day
        self.df["Year_OCC"] = self.df["DATE OCC"].dt.year

        self.df["Date Difference"] = (
            self.df["Date Rptd"] - self.df["DATE OCC"]
        ).dt.days

        self.df["Hour"] = self.df["TIME OCC"] // 100
        self.df["Minute"] = self.df["TIME OCC"] % 100

    def __preprocess_age(self):
        self.df["Vict Age"] = self.df["Vict Age"].apply(lambda x: 0 if x < 0 else x)

    def __preprocess_exotic(self):
        self.df["Vict Sex"] = self.df["Vict Sex"].apply(
            lambda x: x if x in ["M", "F"] else "X"
        )
        self.df["Vict Descent"] = self.df["Vict Descent"].apply(
            lambda x: (
                x
                if x
                in [
                    "A",
                    "B",
                    "C",
                    "D",
                    "F",
                    "G",
                    "H",
                    "I",
                    "J",
                    "K",
                    "L",
                    "O",
                    "P",
                    "S",
                    "U",
                    "V",
                    "W",
                    "X",
                    "Z",
                ]
                else "X"
            )
        )
        self.df["Premis Desc"] = self.df["Premis Desc"].apply(
            lambda x: "UNKNOWN" if pd.isna(x) else x
        )
        self.df["Weapon Desc"] = self.df["Weapon Desc"].apply(
            lambda x: "UNKNOWN" if pd.isna(x) else x
        )
        self.df["Status"] = self.df["Status"].apply(lambda x: "CC" if pd.isna(x) else x)
        self.df["Cross Street"] = self.df["Cross Street"].apply(
            lambda x: "UNKNOWN" if pd.isna(x) else x
        )

    def __preprocess_crime_code(self):
        self.df["Crm Cd"] = self.df["Crm Cd"].astype("category")
        self.df["Label"] = self.df["Crm Cd"].cat.codes

    def __drop_unnecessary_columns(self):
        columns_to_drop = [
            "DR_NO",
            "Date Rptd",
            "DATE OCC",
            "TIME OCC",
            "AREA",
            "Crm Cd",
            "Crm Cd Desc",
            "Mocodes",
            "Premis Cd",
            "Weapon Used Cd",
            "Status Desc",
            "Crm Cd 1",
            "Crm Cd 2",
            "Crm Cd 3",
            "Crm Cd 4",
            "LOCATION",
            "LAT",
            "LON",
        ]

        self.df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    def __save(self, sampled_df=None, remaining_df=None):
        if sampled_df is not None:
            sampled_df.to_csv(os.path.join(BASE_DIR, "crime_labeled.csv"), index=False)
        if remaining_df is not None:
            remaining_df.to_csv(
                os.path.join(BASE_DIR, "crime_support.csv"), index=False
            )
        self.df.to_csv(os.path.join(BASE_DIR, "crime_processed.csv"), index=False)

    def __sample(self):
        sampled_df = self.df.sample(frac=0.1, random_state=42)

        remaining_df = self.df.drop(sampled_df.index)

        remaining_df = remaining_df.drop(columns=["Label"])

        print(sampled_df["Label"].value_counts(normalize=True))

        return sampled_df, remaining_df

    def keep_recent_year_data(self):
        """
        Keep records in 2024.5 ~ 2025.5
        """
        self.df["Date Rptd"] = pd.to_datetime(self.df["Date Rptd"], errors="coerce")
        self.df = self.df[
            (self.df["Date Rptd"] >= "2024-05-01")
            & (self.df["Date Rptd"] <= "2025-05-31")
        ]

    def preprocess(self):
        self.keep_recent_year_data()
        self.__find_top_n(n=10, drop=False)
        self.__preprocess_time()
        self.__preprocess_age()
        self.__preprocess_crime_code()
        self.__preprocess_exotic()
        self.__drop_unnecessary_columns()
        sampled_df, remaining_df = self.__sample()
        self.__save(sampled_df, remaining_df)
        analyse_df_feature(self.df, threshold=1000)

        na_columns = self.df.columns[self.df.isna().any()].tolist()
        if na_columns:
            print(f"Columns have NaN: {na_columns}")
        else:
            print("No column has NaN")

        return self.df


if __name__ == "__main__":
    preprocessor = CrimeDataPreprocessor(
        data_path="../raw/Crime_Data_from_2020_to_Present.csv"
    )
    preprocessor.preprocess()
    print("Crime data preprocessing completed.")
