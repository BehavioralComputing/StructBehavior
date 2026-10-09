"""Align global instance indices with the fixed training labels."""

import torch


def training_selection(split, instance_idx, device, node_num):
    """Return training-only global indices and labels in the same order.

    Graph propagation may include unlabeled context. This selection confines
    supervised examples and contrastive positives/negatives to the training
    instances, including shuffled or non-contiguous dataset partitions.
    """
    train_idx = torch.as_tensor(split["train_idx"], dtype=torch.long, device=device)
    train_labels = torch.as_tensor(split["train_label"], dtype=torch.long, device=device)
    selected = train_idx if instance_idx is None else torch.as_tensor(
        instance_idx, dtype=torch.long, device=device
    )
    if selected.ndim != 1 or selected.numel() == 0:
        raise ValueError("A training mini-batch must contain a nonempty index vector.")
    if train_labels.numel() != train_idx.numel():
        raise ValueError("Training indices and labels must have the same length.")
    if torch.any(selected < 0) or torch.any(selected >= node_num):
        raise ValueError("A mini-batch index is outside the instance population.")
    positions = torch.full((node_num,), -1, dtype=torch.long, device=device)
    positions[train_idx] = torch.arange(train_idx.numel(), device=device)
    selected_positions = positions[selected]
    if torch.any(selected_positions < 0):
        raise ValueError("Contrastive and supervised mini-batches must use training instances only.")
    if selected.unique().numel() != selected.numel():
        raise ValueError("A mini-batch must not repeat instance identities.")
    return selected, train_labels[selected_positions]
