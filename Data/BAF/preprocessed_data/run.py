from sample_data import SampleData
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    in_file = os.path.join(BASE_DIR, "../raw/Base.csv")
    out_path = os.path.join(BASE_DIR, ".")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if os.path.exists(os.path.join(out_path, "baf_labeled.csv")) and os.path.exists(
        os.path.join(out_path, "baf_support.csv")
    ):
        print("Sampled files already exist. Returning without re-sampling.")
        return

    sampler = SampleData(in_file, out_path)
    sampler.sample_with_month(sample_ratio=0.1)
    sampler.sample_with_fixed_ratio(sample_ratio=0.1)
