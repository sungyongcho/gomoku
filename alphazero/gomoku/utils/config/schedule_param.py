from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class SchedulePoint:
    """
    'until'은 1-based 반복 끝(포함)입니다.
    예) until=10 → iteration 1~10에 value 적용.
    """

    until: int
    value: float


type ScheduledFloat = float | list[SchedulePoint]


def _is_points_like(seq: Sequence[Any]) -> bool:
    return bool(seq) and hasattr(seq[0], "until") and hasattr(seq[0], "value")


def _coerce_point(item: Any, *, name: str, index: int) -> SchedulePoint:
    if isinstance(item, SchedulePoint):
        return item

    if isinstance(item, Mapping):
        if "until" not in item or "value" not in item:
            raise ValueError(
                f"[ConfigError] {name}[{index}] requires both 'until' and 'value' keys."
            )
        try:
            until = int(item["until"])
            value = float(item["value"])
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                f"[ConfigError] In {name}[{index}], 'until' must be int and 'value' must be float."
            ) from e
        return SchedulePoint(until=until, value=value)

    if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
        # Accept compact [until, value] or (until, value) pairs for readability.
        if len(item) != 2:
            raise ValueError(
                f"[ConfigError] {name}[{index}] sequence must be length 2 (until, value)."
            )
        try:
            until = int(item[0])
            value = float(item[1])
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                f"[ConfigError] In {name}[{index}], sequence values must be (int until, float value)."
            ) from e
        return SchedulePoint(until=until, value=value)

    raise ValueError(f"[ConfigError] Invalid item format for {name}[{index}]: {item!r}")


def _to_schedule_points(raw_list: Iterable[Any], *, name: str) -> list[SchedulePoint]:
    points: list[SchedulePoint] = []
    last_until = -1

    for i, item in enumerate(raw_list):
        pt = _coerce_point(item, name=name, index=i)

        # allow 0 (no effect for 1-based schedule), but not negative
        if pt.until < 0:
            raise ValueError(
                f"[ConfigError] {name}[{i}].until must be a non-negative integer."
            )

        if pt.until <= last_until:
            raise ValueError(
                f"[ConfigError] 'until' values for {name} must be strictly increasing. "
                f"({last_until} -> {pt.until})"
            )

        last_until = pt.until
        points.append(pt)

    if not points:
        raise ValueError(f"[ConfigError] Schedule for '{name}' cannot be empty.")
    return points


def _validate_schedule_bounds(
    pts: Sequence[SchedulePoint],
    *,
    name: str,
    num_iterations: int,
    require_end_exact: bool,
) -> None:
    for i, pt in enumerate(pts):
        if pt.until > num_iterations:
            raise ValueError(
                f"[ConfigError] {name}[{i}].until ({pt.until}) cannot exceed "
                f"training.num_iterations ({num_iterations})."
            )

    if require_end_exact:
        last_until = pts[-1].until
        if last_until != num_iterations:
            raise ValueError(
                f"[ConfigError] {name} must end exactly at training.num_iterations "
                f"(expected {num_iterations}, got last until={last_until})."
            )


def parse_scheduled_float(
    raw: Any, *, name: str, num_iterations: int
) -> ScheduledFloat:
    """
    Normalize to ScheduledFloat.
    If a schedule list is provided, its last 'until' must be EXACTLY equal to num_iterations.
    """
    if isinstance(raw, (int, float)):
        return float(raw)

    if isinstance(raw, list):
        pts = _to_schedule_points(raw, name=name)
        _validate_schedule_bounds(
            pts, name=name, num_iterations=num_iterations, require_end_exact=True
        )
        return pts

    raise ValueError(f"[ConfigError] Invalid type for {name}: {type(raw).__name__}")


def get_scheduled_value(
    spec: ScheduledFloat | list[Mapping[str, Any]], iteration: int
) -> float:
    """
    Return the effective value for a 0-based iteration.
    For schedules, compare against 1-based 'until'; if exceeded, keep the last value.
    """
    if iteration < 0:
        raise ValueError(
            f"[ConfigError] iteration must be non-negative (got {iteration})."
        )

    if isinstance(spec, (int, float)):
        return float(spec)

    # Allow raw list of dicts as well as SchedulePoint dataclasses
    pts: list[SchedulePoint]
    if _is_points_like(spec):
        pts = spec  # type: ignore[assignment]
    else:
        pts = _to_schedule_points(spec, name="schedule")  # type: ignore[arg-type]

    iter1 = iteration + 1
    untils = [p.until for p in pts]  # strictly increasing by construction
    idx = bisect_left(untils, iter1)
    if idx >= len(pts):
        return float(pts[-1].value)
    return float(pts[idx].value)


def validate_against_num_iterations(
    specs: dict[str, ScheduledFloat | list[Mapping[str, Any]]], *, num_iterations: int
) -> None:
    """
    이미 파싱된(list[SchedulePoint]) 스펙뿐 아니라,
    raw(list[dict]) 스펙도 받아서 '끝 until == num_iterations'를 검증합니다.
    """
    for name, spec in specs.items():
        if isinstance(spec, (int, float)):
            continue
        if not spec:
            raise ValueError(f"[ConfigError] Schedule for '{name}' cannot be empty.")

        pts = spec if _is_points_like(spec) else _to_schedule_points(spec, name=name)  # type: ignore[arg-type]
        _validate_schedule_bounds(
            pts, name=name, num_iterations=num_iterations, require_end_exact=True
        )
