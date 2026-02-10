def calc_num_planes(history_length: int) -> int:
    """Compute required input planes based on history length."""
    return 8 + int(history_length)


# Default NUM_PLANES assumes history_length=5 for backward compatibility.
NUM_PLANES = calc_num_planes(5)

POLICY_CHANNELS = 2
VALUE_CHANNELS = 1


def calc_num_hidden(num_lines: int) -> int:
    """Return a heuristic hidden-channel count based on board size."""
    if num_lines <= 6:
        return 32
    if num_lines <= 12:
        return 64
    return 128


def calc_num_resblocks(num_lines: int) -> int:
    """Return a heuristic residual block count based on board size."""
    if num_lines <= 6:
        return 2
    if num_lines <= 12:
        return 4
    return 6
