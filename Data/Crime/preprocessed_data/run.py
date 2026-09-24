from crime_preprocess import CrimeDataPreprocessor
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    if os.path.exists(os.path.join(BASE_DIR, "crime_labeled.csv")) and os.path.exists(
        os.path.join(BASE_DIR, "crime_support.csv")
    ):
        print("Sampled files already exist. Returning without re-sampling.")
        return

    preprocessor = CrimeDataPreprocessor(
        data_path=os.path.join(BASE_DIR, "../raw/Crime_Data_from_2020_to_Present.csv")
    )
    preprocessor.preprocess()
    print("Crime data preprocessing completed.")
