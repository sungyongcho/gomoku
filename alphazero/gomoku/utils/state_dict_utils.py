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
