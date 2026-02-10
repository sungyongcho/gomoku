import pytest

from gomoku.pvmcts.pvmcts import PVMCTS
from tests.pvmcts.search.conftest import make_game_and_params


class DummyInference:
    def __init__(self, *, async_capable: bool = False, device: str = "cpu"):
        self.device = device
        if async_capable:
            self.infer_async = lambda x: x  # noqa: E731


@pytest.mark.parametrize(
    "inference_device, async_capable, expected",
    [
        ("cpu", False, "sequential"),
        ("cuda", False, "vectorize"),
        ("cpu", True, "ray"),
        ("cuda", True, "ray"),
    ],
)
def test_infer_mode_defaults(inference_device, async_capable, expected):
    game, params = make_game_and_params()
    inference = DummyInference(async_capable=async_capable, device=inference_device)
    mcts = PVMCTS(game, params, inference_client=inference, mode=None)
    assert mcts.mode == expected


@pytest.mark.parametrize("mode", ["sequential", "vectorize", "mp", "ray"])
def test_mode_accepts_supported(mode: str):
    game, params = make_game_and_params()
    inference = DummyInference(device="cpu")
    mcts = PVMCTS(game, params, inference_client=inference, mode=mode)
    assert mcts.mode == mode


def test_mode_rejects_unknown():
    game, params = make_game_and_params()
    inference = DummyInference(device="cpu")
    with pytest.raises(ValueError):
        PVMCTS(game, params, inference_client=inference, mode="unknown")
