import pandas as pd


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
