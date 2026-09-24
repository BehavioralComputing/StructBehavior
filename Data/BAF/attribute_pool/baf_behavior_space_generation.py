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


class TwiBehavior:
    def __init__(self, root="./", device="cuda:0", mid_generated=False):
        self.root = root
        self.device = device
        self.mid_generated = mid_generated

        if not mid_generated:
            self.data_df = pd.read_csv(
                os.path.join(self.root, "baf_preprocessed_data_second_step.csv"),
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
        basic_info = [
            "income",
            "employment_status",
            "customer_age",
            "housing_status",
            "bank_months_count",
            "has_other_cards",
            "date_of_birth_distinct_emails_4w",
        ]
        geo_address = [
            "prev_address_months_count",
            "current_address_months_count",
            "zip_count_4w",
        ]
        device_info = [
            "device_os",
            "device_distinct_emails_8w",
            "device_fraud_count",
            "source",
            "session_length_in_minutes",
            "keep_alive_session",
        ]
        contact_info = [
            "name_email_similarity",
            "email_is_free",
            "phone_home_valid",
            "phone_mobile_valid",
        ]
        velocity = [
            "velocity_6h",
            "velocity_24h",
            "velocity_4w",
            "bank_branch_count_8w",
            "foreign_request",
        ]
        credit_intent = [
            "proposed_credit_limit",
            "intended_balcon_amount",
            "payment_type",
            "credit_risk_score",
        ]
        time_info = ["days_since_request", "month"]

        complete_link = [
            basic_info,
            geo_address,
            device_info,
            contact_info,
            velocity,
            credit_intent,
            time_info,
        ]

        single_link = [
            ["bank_months_count", "proposed_credit_limit"],
            ["zip_count_4w", "velocity_4w"],
            ["device_os", "velocity_6h"],
            ["email_is_free", "date_of_birth_distinct_emails_4w"],
            ["bank_months_count", "velocity_24h"],
            ["current_address_months_count", "phone_home_valid"],
            ["days_since_request", "source"],
        ]

        self.graph_all = single_link + complete_link
        print("Graph all:", self.graph_all)

    def __attr_val_to_index(self):
        if not self.mid_generated:
            for p_col in self.data_df.columns:
                print("Handling: ", p_col)

                unique_values = self.data_df[p_col].unique()
                print("length of unique values: ", len(unique_values))

                encoding = {value: i for i, value in enumerate(unique_values)}

                self.column_encoding[p_col] = encoding

            for col, enc in self.column_encoding.items():
                self.data_df[col] = self.data_df[col].map(enc)

            self.__save_mid_process()
        else:
            with open(os.path.join(BASE_DIR, "./baf_column_encoding.json"), "r") as f:
                self.column_encoding = json.load(f)
            print("baf_column_encoding.json loaded")

            self.data_df = pd.read_csv(os.path.join(BASE_DIR, "./baf_data_df.csv"))

            print("baf_data_df.csv loaded")

    def __save_mid_process(self):
        with open(os.path.join(BASE_DIR, "./baf_column_encoding.json"), "w") as f:
            json.dump(self.column_encoding, f)
        print("baf_column_encoding.json saved")

        self.data_df.to_csv(os.path.join(BASE_DIR, "./baf_data_df.csv"), index=False)
        print("baf_data_df.csv saved")

    def __connect_graph(self):
        data_dict = {}

        for graph in self.graph_all:
            complete_graph_edge(data_dict, graph, self.data_df)

        self.hetero_graph = dgl.heterograph(data_dict).to(self.device)

        print("Graph connected")

    def __decorate_text(self, col: str, val: str):
        if col == "income":
            return f"Annual income of the applicant is {val}."
        elif col == "name_email_similarity":
            return f"The similarity between the applicant's name and email is {val}."
        elif col == "prev_address_months_count":
            return (
                f"The applicant has lived at their previous residence for {val} months."
            )
        elif col == "current_address_months_count":
            return (
                f"The applicant has lived at their current residence for {val} months."
            )
        elif col == "customer_age":
            return f"The applicant's is {val} years old."
        elif col == "days_since_request":
            return f"{val} days have passed since the application was done."
        elif col == "intended_balcon_amount":
            return f"The applicant intends to initially transfer {val} to the account."
        elif col == "payment_type":
            return f"The applicant's preferred credit payment type is {val}."
        elif col == "zip_count_4w":
            return f"There are {val} applications within the same zip code in last 4 weeks."
        elif col == "velocity_6h":
            return f"There are {val} applications made per hour in the last 6 hours."
        elif col == "velocity_24h":
            return f"There are {val} applications made per hour in the last 24 hours."
        elif col == "velocity_4w":
            return f"There are {val} applications made per hour in the last 4 weeks."
        elif col == "bank_branch_count_8w":
            return f"There are {val} total applications in the selected bank branch in last 8 weeks."
        elif col == "date_of_birth_distinct_emails_4w":
            return f"There are {val} emails for applicants with the same date of birth in last 4 weeks."
        elif col == "employment_status":
            return f"The applicant's employment status is {val}."
        elif col == "credit_risk_score":
            return f"The applicant's credit risk score is {val}."
        elif col == "email_is_free":
            if val == "1":
                return "The applicant is using a free email service provider."
            elif val == "0":
                return "The applicant is using a paid email service provider."
            else:
                raise ValueError(f"Unknown value for email_is_free: {val}")
        elif col == "housing_status":
            return f"The applicant's housing status is {val}."
        elif col == "phone_home_valid":
            if val == "1":
                return "The applicant's home phone number is valid."
            elif val == "0":
                return "The applicant's home phone number is invalid."
            else:
                raise ValueError(f"Unknown value for phone_home_valid: {val}")
        elif col == "phone_mobile_valid":
            if val == "1":
                return "The applicant's mobile phone number is valid."
            elif val == "0":
                return "The applicant's mobile phone number is invalid."
            else:
                raise ValueError(f"Unknown value for phone_mobile_valid: {val}")
        elif col == "bank_months_count":
            return f"The previous bank account of the applicant has been active for {val} months."
        elif col == "has_other_cards":
            if val == "1":
                return "The applicant has other credit cards from the same banking company."
            elif val == "0":
                return "The applicant does not have other credit cards from the same banking company."
            else:
                raise ValueError(f"Unknown value for has_other_cards: {val}")
        elif col == "proposed_credit_limit":
            return f"The applicant has proposed a credit limit of {val}."
        elif col == "foreign_request":
            if val == "1":
                return "The origin country of request is different from bank's country."
            elif val == "0":
                return "The origin country of request is the same as bank's country."
            else:
                raise ValueError(f"Unknown value for foreign_request: {val}")
        elif col == "source":
            return f"The source of the request is {val}."
        elif col == "session_length_in_minutes":
            return f"The length of user session in banking website is {val} minutes."
        elif col == "device_os":
            return f"The operating system of the device that made the request is {val}."
        elif col == "keep_alive_session":
            if val == "1":
                return "The user has kept the session alive in the banking website."
            elif val == "0":
                return "The user has not kept the session alive in the banking website."
            else:
                raise ValueError(f"Unknown value for keep_alive_session: {val}")
        elif col == "device_distinct_emails_8w":
            return f"There are {val} distinct emails in banking website from the used device in last 8 weeks."
        elif col == "device_fraud_count":
            return f"There are {val} fraudulent applications with the used device"
        elif col == "month":
            month_map = {
                "0": "February",
                "1": "March",
                "2": "April",
                "3": "May",
                "4": "June",
                "5": "July",
                "6": "August",
                "7": "September",
            }
            return f"The application was made in {month_map.get(val, 'Unknown month')}."
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

            print("Feature embedding for ", col, " generated")

        dgl.save_graphs(
            os.path.join(BASE_DIR, "./baf_Data_hetero_graph.dgl"), [self.hetero_graph]
        )
        print(self.hetero_graph)


if __name__ == "__main__":
    twi = TwiBehavior(mid_generated=False)
