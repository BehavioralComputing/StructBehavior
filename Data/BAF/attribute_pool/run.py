from baf_preprocess_first_step import PreprocessFirstStep
from baf_preprocess_second_step import PreprocessSecondStep
from baf_behavior_space_generation import TwiBehavior
from baf_weighted_edge import WeightedEdge
import dgl
import os


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(
        os.path.join(BASE_DIR, "baf_preprocessed_data_first_step.csv")
    ):
        preprocess_first_step = PreprocessFirstStep(
            root=os.path.join(BASE_DIR, "../preprocessed_data/")
        )
        preprocess_first_step.preprocess()
        preprocess_first_step.save(
            os.path.join(BASE_DIR, "baf_preprocessed_data_first_step.csv")
        )
    else:
        print("First step preprocessed data already exists. Skipping first step.")

    if not os.path.exists(
        os.path.join(BASE_DIR, "baf_preprocessed_data_second_step.csv")
    ):
        preprocess_second_step = PreprocessSecondStep(root=BASE_DIR)
        preprocess_second_step.preprocess()
    else:
        print("Second step preprocessed data already exists. Skipping second step.")

    if not os.path.exists(os.path.join(BASE_DIR, "baf_Data_hetero_graph.dgl")):
        TwiBehavior(root=BASE_DIR)
    else:
        print("Behavior space data already exists. Skipping behavior space generation.")

    if not os.path.exists(os.path.join(BASE_DIR, "baf_weighted_graph.dgl")):
        g = dgl.load_graphs(os.path.join(BASE_DIR, "baf_Data_hetero_graph.dgl"))[0][0]
        weighted_edge = WeightedEdge(g)
        weighted_edge.run()
    else:
        print("Weighted edge data already exists. Skipping weighted edge generation.")

    print("All steps completed successfully.")
