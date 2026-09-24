from datetime import datetime
import pandas as pd


def two_significant_digits(num_str, debug_msg=False):
    if num_str == "None ":
        print("None!")
        return "None "
    elif pd.isna(num_str):
        return None

    try:
        num = int(num_str)
    except Exception as e:
        print(f"Invalid input: {num_str} is not an integer")
        if debug_msg:
            input("Press Enter to continue...")
        return None

    digits = len(str(abs(num)))

    if digits <= 2:
        return num

    zeros = digits - 2

    rounded = round(num / (10**zeros)) * (10**zeros)

    if debug_msg:
        print(f"Original number: {num_str}, Rounded number: {rounded}")

    return rounded


def decompose_date(date_str: str, debug_msg=False):
    if date_str == "None ":
        print("None!")
        return "None "
    elif pd.isna(date_str):
        return {
            "year": None,
            "month": None,
            "day": None,
            "hour": None,
            "minute": None,
            "second": None,
        }

    date_format = "%a %b %d %H:%M:%S %z %Y"

    try:
        date_obj = datetime.strptime(date_str.strip(), date_format)
    except Exception as e:
        print(f"Invalid input: {date_str} is not a valid date string")
        if debug_msg:
            input("Press Enter to continue...")
        return {
            "year": None,
            "month": None,
            "day": None,
            "hour": None,
            "minute": None,
            "second": None,
        }

    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    hour = date_obj.hour
    minute = date_obj.minute
    second = date_obj.second

    if debug_msg:
        print(f"Parsed date: {date_obj}")
        print(
            f"Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Minute: {minute}, Second: {second}"
        )

    return {
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "second": second,
    }


def decompose_color(rgb_str: str, debug_msg=False):
    if rgb_str == "None ":
        print("None!")
        return "None "
    elif pd.isna(rgb_str):
        return {"red": None, "green": None, "blue": None}

    try:
        rgb_str1 = rgb_str.strip()
    except Exception as e:
        print(f"Invalid input: {rgb_str} is not a valid RGB string")
        if debug_msg:
            input("Press Enter to continue...")
        return {"red": None, "green": None, "blue": None}

    if len(rgb_str1) != 6:
        print(f"Invalid input: {rgb_str1} is not a valid RGB string")
        if debug_msg:
            input("Press Enter to continue...")
        return {"red": None, "green": None, "blue": None}

    r = int(rgb_str1[0:2], 16)
    g = int(rgb_str1[2:4], 16)
    b = int(rgb_str1[4:6], 16)

    if debug_msg:
        print(f"Parsed RGB: {rgb_str1} -> Red: {r}, Green: {g}, Blue: {b}")

    return {"red": r, "green": g, "blue": b}


def str_list_to_bool_dict(list_in: list, std_list: list, debug_msg=False):
    if list_in == "None ":
        print("None!")
        return "None "

    if not isinstance(list_in, list):
        print(f"Invalid input: {list_in} is not a list")
        if debug_msg:
            input("Press Enter to continue...")
        return {item: None for item in std_list}

    bool_dict = {item: item in list_in for item in std_list}

    if debug_msg:
        print(f"Converted {list_in} to boolean dictionary: {bool_dict}")

    return bool_dict


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

    print(two_significant_digits("None "))
    print(two_significant_digits(float("nan")))
    print(two_significant_digits(pd.NA))

    print(decompose_date("None "))
    print(decompose_date(float("nan")))

    print(decompose_color("None "))
    print(decompose_color(float("nan")))
    print(decompose_color(pd.NA))

    print(str_list_to_bool_dict("None ", ["item1", "item2", "item3"]))
    print(str_list_to_bool_dict(float("nan"), ["item1", "item2", "item3"]))
    print(str_list_to_bool_dict(pd.NA), ["item1", "item2", "item3"])
