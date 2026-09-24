from preprocess_instafake import PreprocessInstafake
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    input_dir = os.path.join(BASE_DIR, "../raw")
    output_dir = os.path.join(BASE_DIR, ".")

    preprocessor = PreprocessInstafake(input_dir, output_dir)
    preprocessor.preprocess()
    print("Instafake data preprocessing completed.")
