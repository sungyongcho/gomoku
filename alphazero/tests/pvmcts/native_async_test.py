"""Native search through the asynchronous inference path used by Ray inference actors."""

import numpy as np
import pytest
import torch

ray = pytest.importorskip("ray")

from gomoku.core.gomoku import Gomoku
from gomoku.pvmcts.pvmcts import PVMCTS
from gomoku.utils.config.loader import BoardConfig, MctsConfig

SIZE = 9


@pytest.fixture(scope="module", autouse=True)
def _ray():
    ray.init(num_cpus=1, include_dashboard=False, ignore_reinit_error=True)
    yield
    ray.shutdown()


class _AsyncInference:
    """Returns uniform bfloat16 outputs, as a CPU inference actor under autocast does."""

    device = torch.device("cpu")

    def __init__(self, broken: bool = False) -> None:
        self.broken = broken

    def infer(self, states, native_payload=None, model_slot=None):
        raise AssertionError("the batched native search must use infer_async")

    def infer_async(self, states, native_payload=None, model_slot=None):
        if self.broken:
            return ray.put("not a (policy, value) pair")
        batch = states.shape[0] if states.dim() == 4 else 1
        policy = torch.zeros(batch, SIZE * SIZE, dtype=torch.bfloat16)
        value = torch.zeros(batch, 1, dtype=torch.bfloat16)
        return ray.put((policy, value))


def _search(inference: _AsyncInference, add_noise: bool) -> list[float]:
    board = BoardConfig(
        num_lines=SIZE,
        enable_doublethree=False,
        enable_capture=False,
        capture_goal=5,
        gomoku_goal=5,
        history_length=1,
    )
    game = Gomoku(board, use_native=True)
    params = MctsConfig(
        C=2.0,
        num_searches=24,
        exploration_turns=0,
        dirichlet_epsilon=0.5,
        dirichlet_alpha=0.03,
        batch_infer_size=8,
        max_batch_wait_ms=0,
        min_batch_size=1,
        use_native=True,
        fpu_from_parent=True,
    )
    search = PVMCTS(game, params, inference, mode="sequential")
    np.random.seed(0)
    torch.manual_seed(0)
    search.run_search([search.create_root(game.get_initial_state())], add_noise=add_noise)
    return [prior for _, prior in search.engine._native_engine.root_priors()]


def test_async_search_reads_bfloat16_outputs_and_mixes_root_noise() -> None:
    priors = _search(_AsyncInference(), add_noise=True)
    assert max(priors) > 10 * (1 / len(priors))


def test_async_search_without_noise_keeps_the_uniform_root() -> None:
    priors = _search(_AsyncInference(), add_noise=False)
    assert max(priors) == pytest.approx(min(priors))


def test_async_inference_errors_stop_the_search_instead_of_hanging() -> None:
    with pytest.raises(Exception):
        _search(_AsyncInference(broken=True), add_noise=False)
