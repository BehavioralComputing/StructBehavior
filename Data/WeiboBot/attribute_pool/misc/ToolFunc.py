from datetime import datetime
import pandas as pd
from math import floor, log10


def two_significant_digits_strict(num_in, debug_msg=False):
    if pd.isna(num_in):
        return None

    try:
        num = float(str(num_in).strip())
    except Exception as e:
        print(f"Invalid input: {num_in} is not a valid number")
        if debug_msg:
            input("Press Enter to continue...")
        return None

    if num == 0:
        return 0

    sign = -1 if num < 0 else 1
    num = abs(num)

    magnitude = floor(log10(num))
    normalized = num / (10**magnitude)
    rounded = round(normalized * 10) / 10
    result = sign * rounded * (10**magnitude)

    result = round(result, 2)

    if isinstance(num_in, int):
        result = int(result)

    if debug_msg:
        print(f"Original number: {num_in}, Rounded number: {result}")

    return result


def analyse_csv_feature(file_path: str, threshold: int = 1000):

    df = pd.read_csv(file_path)
    count = 0
    max_col_length = max(len(str(col)) for col in df.columns)

    for column in df.columns:
        n_unique = df[column].nunique()
        if n_unique < threshold:
            print(
                f"{column:<{max_col_length}}  |  dtype: {str(df[column].dtype):<8}  |  unique: {n_unique}"
            )
        else:
            print(
                f"【TOO MANY】{column:<{max_col_length}}  |  dtype: {str(df[column].dtype):<8}  |  unique: {n_unique}"
            )
        count += n_unique

    print(f"\nTotal unique values: {count}")


def analyse_df_feature(df: pd.DataFrame, threshold: int = 1000):
    count = 0

    max_col_length = max(len(str(col)) for col in df.columns)

    for column in df.columns:
        try:
            n_unique = df[column].nunique()
        except Exception as e:
            n_unique = df[column].apply(lambda x: str(x)).nunique()
        if n_unique < threshold:
            print(
                f"{column:<{max_col_length}}  |  dtype: {str(df[column].dtype):<8}  |  unique: {n_unique}"
            )
        else:
            print(
                f"{column:<{max_col_length}}  |  dtype: {str(df[column].dtype):<8}  |  unique: {n_unique} 【TOO MANY】"
            )
        count += n_unique

    print(f"\nTotal unique values: {count}")


if __name__ == "__main__":

    pass
