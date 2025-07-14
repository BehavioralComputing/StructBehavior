import argparse
import os
import importlib.util
import sys

def run_stage(script_path):
    stage_dir = os.path.dirname(script_path)
    
    if stage_dir not in sys.path:
        sys.path.insert(0, stage_dir)
    
    spec = importlib.util.spec_from_file_location("stage_module", script_path)
    stage_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage_module)

    if hasattr(stage_module, "main"):
        stage_module.main()
    else:
        print(f"⚠️ {script_path} doesn't define main().")

    if stage_dir in sys.path:
        sys.path.remove(stage_dir)

def main(dataset):
    dataset_dir = dataset
    if not os.path.isdir(dataset_dir):
        valid_datasets = [d for d in os.listdir('.') if os.path.isdir(d)]
        print(f"❌ Dataset {dataset} does not exist. Valid datasets are: {valid_datasets}")
        return

    stages = ['preprocessed_data', 'material_data', 'attribute_pool', 'attribute_tree']

    for stage in stages:
        script_path = os.path.join(dataset_dir, stage, "run.py")
        if os.path.exists(script_path):
            print(f"▶️ Now executing {stage} stage: {script_path}")
            run_stage(script_path)
        else:
            print(f"⏭️ Skip {stage} stage(No script found)")

    print("✅ All stages completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universial Data Processing Pipeline")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name(e.g. "BAF")')
    args = parser.parse_args()
    main(args.dataset)
