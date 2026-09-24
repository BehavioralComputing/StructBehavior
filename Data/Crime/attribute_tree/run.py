from crime_attribute_tree import Crime_attribute_tree


def main():
    dataset = Crime_attribute_tree(device="cuda:0", process=True, save=True)
    dataset.attribute_graph_generate()
    dataset.attribute_tree_generate()
