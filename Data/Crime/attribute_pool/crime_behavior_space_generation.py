import pandas as pd
import dgl
import torch
import math
import numpy as np
from transformers import BertTokenizer, BertModel
import json
from datetime import datetime as dt
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def del_nan(A: list, B: list):
    array_A = np.array(A)
    array_B = np.array(B)
    nan_mask = np.isnan(array_A) | np.isnan(array_B)
    valid_indices = ~nan_mask
    return array_A[valid_indices], array_B[valid_indices]


def complete_graph_edge(
    data_dict: dict, columns_to_convert: list, data_df: pd.DataFrame
):
    for src in columns_to_convert:
        for tar in columns_to_convert:
            if src != tar:
                new_list_A, new_list_B = del_nan(data_df[src], data_df[tar])
                data_dict.update(
                    {
                        (src, "edge_" + src + "_" + tar, tar): (
                            torch.tensor(new_list_A, dtype=torch.int),
                            torch.tensor(new_list_B, dtype=torch.int),
                        )
                    }
                )


class CrimeBehavior:
    def __init__(self, root="./", device="cuda:0", mid_generated=False):
        self.root = root
        self.device = device
        self.mid_generated = mid_generated

        if not mid_generated:
            self.data_df = pd.read_csv(
                os.path.join(self.root, "crime_preprocessed_data_second_step.csv"),
                encoding="utf-8",
                dtype="string",
            )

        self.column_encoding = {}
        self.graph_all = []
        self.hetero_graph = None

        self.__get_graph_all()
        self.__attr_val_to_index()
        self.__connect_graph()
        self.__gen_embedding()

    def __get_graph_all(self):
        tim_pla_peo_weap = [
            "Vict Age",
            "Vict Sex",
            "Vict Descent",
            "Premis Desc",
            "Weapon Desc",
            "Month_OCC",
            "Day_OCC",
            "Year_OCC",
            "Hour",
            "Minute",
        ]

        pcl = ["Premis Desc", "Cross Street"]

        rpa = ["Rpt Dist No", "Premis Desc", "AREA NAME"]

        rsp = ["Status", "Part 1-2", "Rpt Dist No"]

        rdp = ["Day_Rptd", "Rpt Dist No", "Premis Desc"]

        dym = ["Month_Rptd", "Day_Rptd", "Year_Rptd"]

        ddd = ["Date Difference", "Day_Rptd", "Day_OCC"]

        self.graph_all = [tim_pla_peo_weap, pcl, rpa, rsp, rdp, dym, ddd]

    def __attr_val_to_index(self):
        if not self.mid_generated:
            for p_col in self.data_df.columns:

                unique_values = self.data_df[p_col].unique()

                encoding = {value: i for i, value in enumerate(unique_values)}

                self.column_encoding[p_col] = encoding

            for col, enc in self.column_encoding.items():
                self.data_df[col] = self.data_df[col].map(enc)

            self.__save_mid_process()
        else:
            with open(os.path.join(BASE_DIR, "./crime_column_encoding.json"), "r") as f:
                self.column_encoding = json.load(f)
            print("crime_column_encoding.json loaded")

            self.data_df = pd.read_csv(os.path.join(BASE_DIR, "./crime_data_df.csv"))

            print("crime_data_df.csv loaded")

    def __save_mid_process(self):
        with open(os.path.join(BASE_DIR, "./crime_column_encoding.json"), "w") as f:
            json.dump(self.column_encoding, f)
        print("crime_column_encoding.json saved")

        self.data_df.to_csv(os.path.join(BASE_DIR, "./crime_data_df.csv"), index=False)
        print("crime_data_df.csv saved")

    def __connect_graph(self):
        data_dict = {}

        for graph in self.graph_all:
            complete_graph_edge(data_dict, graph, self.data_df)

        self.hetero_graph = dgl.heterograph(data_dict).to(self.device)

        print("Graph connected")

    def __decorate_text(self, col: str, val: str):
        if col == "AREA NAME":
            return f"The crime occurred in the area named {val}."
        elif col == "Rpt Dist No":
            return f"The report district number is {val}."
        elif col == "Part 1-2":
            return f"The crime is classified as part {val} crime."
        elif col == "Vict Age":
            if val == "Unknown":
                return "The age of the victim is unknown."
            else:
                return f"The victim is {val} years old."
        elif col == "Vict Sex":
            if val == "X":
                return "The victim's sex is unknown."
            elif val == "F":
                return "The victim is female."
            elif val == "M":
                return "The victim is male."
            else:
                raise ValueError(f"Unknown value for Vict Sex: {val}")
        elif col == "Vict Descent":
            corres_dict = {
                "A": "Other Asian",
                "B": "Black",
                "C": "Chinese",
                "D": "Cambodian",
                "F": "Filipino",
                "G": "Guamanian",
                "H": "Hispanic or Latin or Mexican",
                "I": "American Indian or Alaskan Native",
                "J": "Japanese",
                "K": "Korean",
                "L": "Laotian",
                "O": "Other",
                "P": "Pacific Islander",
                "S": "Samoan",
                "U": "Hawaiian",
                "V": "Vietnamese",
                "W": "White",
                "X": "Unknown",
                "Z": "Asian Indian",
            }

            if val in corres_dict:
                return f"The victim's descent is {corres_dict[val]}."
            else:
                raise ValueError(f"Unknown value for Vict Descent: {val}")
        elif col == "Premis Desc":
            return f"The crime occurred in a {val}."
        elif col == "Weapon Desc":
            return f"The weapon used in the crime was {val}."
        elif col == "Status":
            corres_dict = {
                "IC": "Investigation Continuing",
                "AO": "Adult Other",
                "AA": "Adult Arrest",
                "JO": "Juvenile Other",
                "JA": "Juvenile Arrest",
                "CC": "Unknown",
            }

            if val in corres_dict:
                return f"The status of the crime is {corres_dict[val]}."
            else:
                raise ValueError(f"Unknown value for Status: {val}")
        elif col == "Cross Street":
            return f"The crime occurred at {val} cross street."
        elif col == "Month_Rptd":
            corres_dict = {
                "1": "January",
                "2": "February",
                "3": "March",
                "4": "April",
                "5": "May",
                "6": "June",
                "7": "July",
                "8": "August",
                "9": "September",
                "10": "October",
                "11": "November",
                "12": "December",
            }
            if val in corres_dict:
                return f"The crime was reported in {corres_dict[val]}."
            else:
                raise ValueError(f"Unknown value for Month_Rptd: {val}")
        elif col == "Day_Rptd":
            return f"The crime was reported on the {val}th day of the month."
        elif col == "Year_Rptd":
            return f"The crime was reported in the year {val}."
        elif col == "Month_OCC":
            corres_dict = {
                "1": "January",
                "2": "February",
                "3": "March",
                "4": "April",
                "5": "May",
                "6": "June",
                "7": "July",
                "8": "August",
                "9": "September",
                "10": "October",
                "11": "November",
                "12": "December",
            }
            if val in corres_dict:
                return f"The crime occurred in {corres_dict[val]}."
            else:
                raise ValueError(f"Unknown value for Month_OCC: {val}")
        elif col == "Day_OCC":
            return f"The crime occurred on the {val}th day of the month."
        elif col == "Year_OCC":
            return f"The crime occurred in the year {val}."
        elif col == "Date Difference":
            return (
                f"{val} days passed between the report and the occurrence of the crime."
            )
        elif col == "Hour":
            return f"The crime occurred at {val} o'clock."
        elif col == "Minute":
            return f"The crime occurred at {val} minutes past the hour."
        else:
            raise ValueError(f"Unknown column: {col}")

    def __gen_embedding(self):
        for col in self.column_encoding.keys():
            print("Now processing feature: ", col)
            feature_list = self.column_encoding[col].keys()
            feature_list_strings = list(map(str, feature_list))

            model_name = f"{os.path.join(BASE_DIR, '../../../bert/bert-base-uncased')}"
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
            model.to(self.device)
            semantic_vectors = []
            feature_len = len(feature_list_strings)
            counter = 0
            for value in feature_list_strings:

                counter += 1
                if counter % 100 == 0:
                    print(f"Processing {counter}/{feature_len} for {col}...")

                value = self.__decorate_text(col, value)

                encoded_text = tokenizer(
                    value,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                with torch.no_grad():
                    outputs = model(**encoded_text)
                    semantic_vector = outputs.last_hidden_state.mean(dim=1).to(
                        self.device
                    )
                    semantic_vectors.append(semantic_vector)
            self.hetero_graph.nodes[col].data["feat"] = torch.cat(
                semantic_vectors, dim=0
            ).to(self.device)

        dgl.save_graphs(
            os.path.join(BASE_DIR, "./crime_Data_hetero_graph.dgl"), [self.hetero_graph]
        )


if __name__ == "__main__":
    crime = CrimeBehavior(mid_generated=False)
