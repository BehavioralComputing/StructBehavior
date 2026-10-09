"""Node-shared relation coefficients for the molecular encoders."""

import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn


class _WeightedRelation(nn.Module):
    def __init__(self, convolution):
        super().__init__()
        self.convolution = convolution

    def set_allow_zero_in_degree(self, value):
        # Preserve HeteroGraphConv's handling of empty incoming neighborhoods.
        if hasattr(self.convolution, "set_allow_zero_in_degree"):
            self.convolution.set_allow_zero_in_degree(value)

    def forward(self, graph, features, relation_coefficient, **kwargs):
        return relation_coefficient * self.convolution(graph, features, **kwargs)


class SharedRelationHeteroConv(nn.Module):
    """Sum relation messages using one learned softmax per view and layer.

    Coefficients normalize over all configured relations and are shared by all
    nodes and feature coordinates. Explicit relation keys avoid dependence on
    the ordering in which DGL aggregates incoming relation outputs.
    """

    def __init__(self, modules, aggregate="sum"):
        super().__init__()
        if aggregate != "sum":
            raise ValueError("Shared relation coefficients require sum aggregation.")
        self.relations = tuple(modules)
        if not self.relations:
            raise ValueError("At least one relation is required.")
        self.relation_logits = nn.Parameter(torch.zeros(len(self.relations)))
        self.convolution = dglnn.HeteroGraphConv(
            {rel: _WeightedRelation(module) for rel, module in modules.items()},
            aggregate="sum",
        )

    def relation_coefficients(self):
        return torch.softmax(self.relation_logits, dim=0)

    def forward(self, graph, node_features, mod_kwargs=None):
        supplied_kwargs = mod_kwargs or {}
        coefficients = self.relation_coefficients()
        relation_kwargs = {}
        for index, relation in enumerate(self.relations):
            relation_kwargs[relation] = {
                **supplied_kwargs.get(relation, {}),
                "relation_coefficient": coefficients[index],
            }
        return self.convolution(graph, node_features, mod_kwargs=relation_kwargs)
