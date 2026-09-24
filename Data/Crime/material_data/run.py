from crime_pt_generator import CRIMEPtGenerator
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    data_dir = os.path.join(BASE_DIR, "../preprocessed_data/")
    output_dir = os.path.join(BASE_DIR, ".")
    generator = CRIMEPtGenerator(data_dir, output_dir)
    generator.run()
