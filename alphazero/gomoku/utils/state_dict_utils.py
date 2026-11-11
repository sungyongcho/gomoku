import torch


def _resize_tensor(source: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Zero-pad the source tensor to match the target shape."""
    resized = torch.zeros(target_shape, dtype=source.dtype, device=source.device)
    slices = tuple(slice(0, min(s, t)) for s, t in zip(source.shape, target_shape))
    resized[slices] = source[slices]
    return resized


def align_state_dict_to_model(
    state_dict: dict[str, torch.Tensor], model_state: dict[str, torch.Tensor]
) -> tuple[list[str], list[str], list[str]]:
    """
    Adjusts checkpoint tensors so they can be loaded into a (usually larger) model.

    Returns (resized_keys, missing_keys, dropped_keys).
    """
    resized_keys: list[str] = []

    for key, target_tensor in model_state.items():
        if key not in state_dict:
            continue
        tensor = state_dict[key]
        if tensor.shape == target_tensor.shape:
            continue
        if tensor.ndim != target_tensor.ndim:
            continue
        state_dict[key] = _resize_tensor(tensor, target_tensor.shape)
        resized_keys.append(key)

    missing_keys = [k for k in model_state.keys() if k not in state_dict]
    dropped_keys = [k for k in list(state_dict.keys()) if k not in model_state]
    for key in dropped_keys:
        state_dict.pop(key, None)

    return resized_keys, missing_keys, dropped_keys


def average_state_dicts(
    states: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Mean of the floating-point tensors; other tensors come from the last state.

    Late in a run the weights wander around a good region from one iteration to the
    next, and their mean is often stronger than any single checkpoint. Batch-norm
    running statistics are averaged with the weights; integer buffers such as batch
    counters are taken from the last state.
    """
    if not states:
        raise ValueError("no checkpoints to average")
    keys = list(states[-1])
    for state in states:
        if list(state) != keys:
            raise ValueError("checkpoints do not share the same tensors")
    averaged = {}
    for key in keys:
        last = states[-1][key]
        if last.is_floating_point():
            stacked = torch.stack([state[key].to(torch.float64) for state in states])
            averaged[key] = stacked.mean(dim=0).to(last.dtype)
        else:
            averaged[key] = last.clone()
    return averaged
