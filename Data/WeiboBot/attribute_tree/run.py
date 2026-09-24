from weibo_attribute_tree import Weibo_attribute_tree


def main():
    dataset = Weibo_attribute_tree(device="cuda:0", process=True, save=True)
    dataset.attribute_graph_generate()
    dataset.attribute_tree_generate()
