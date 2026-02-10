"""Helper for consistent tqdm progress bars."""

from typing import Any

from tqdm import tqdm


def make_progress(
    *,
    total: int | float | None,
    desc: str | None,
    unit: str = "it",
    disable: bool = False,
    leave: bool = True,
    **kwargs: Any,
) -> tqdm:
    """Create a standardized tqdm progress bar."""
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        disable=disable or total is None or total == 0,
        **kwargs,
    )
