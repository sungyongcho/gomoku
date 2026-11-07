"""LocalInference dtype option used by self-play workers that run the model on their own CPU."""

import pytest
import torch

from gomoku.core.gomoku import Gomoku
from gomoku.inference.local import LocalInference
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import BoardConfig, InferenceRuntimeConfig, ModelConfig

SIZE = 9


def _board() -> BoardConfig:
    return BoardConfig(
        num_lines=SIZE,
        enable_doublethree=True,
        enable_capture=True,
        capture_goal=5,
        gomoku_goal=5,
        history_length=5,
    )


def _model() -> tuple[Gomoku, PolicyValueNet]:
    game = Gomoku(_board())
    planes = int(game.get_encoded_state(game.get_initial_state()).shape[1])
    torch.manual_seed(0)
    config = ModelConfig(num_planes=planes, num_hidden=16, num_resblocks=1)
    return game, PolicyValueNet(game, config, device="cpu")


def test_bfloat16_returns_float32_close_to_float32():
    game, model = _model()
    states = torch.as_tensor(
        game.get_encoded_state([game.get_initial_state()] * 4), dtype=torch.float32
    )
    ref_policy, ref_value = LocalInference(model).infer(states)
    policy, value = LocalInference(model, dtype="bfloat16").infer(states)

    assert policy.dtype == torch.float32 and value.dtype == torch.float32
    assert policy.shape == ref_policy.shape and value.shape == ref_value.shape
    assert torch.allclose(policy, ref_policy, atol=0.1)
    assert torch.allclose(value, ref_value, atol=0.05)


def test_unknown_dtype_is_rejected():
    _, model = _model()
    with pytest.raises(ValueError):
        LocalInference(model, dtype="float16")


def test_config_defaults_to_float32():
    assert InferenceRuntimeConfig().local_dtype == "float32"
    assert InferenceRuntimeConfig(local_dtype="bfloat16").local_dtype == "bfloat16"


def test_native_search_calls_local_inference_synchronously():
    from gomoku.pvmcts.pvmcts import PVMCTS
    from gomoku.utils.config.loader import MctsConfig

    _, model = _model()
    game = Gomoku(_board(), use_native=True)

    class SyncOnly(LocalInference):
        def infer_async(self, *args, **kwargs):
            raise AssertionError("local inference must not go through the Ray object store")

    mcts = MctsConfig(
        C=2.0, num_searches=32, exploration_turns=0, dirichlet_epsilon=0.0, dirichlet_alpha=0.03,
        batch_infer_size=8, min_batch_size=1, max_batch_wait_ms=0, use_native=True,
    )
    policy = PVMCTS(game, mcts, SyncOnly(model), mode="ray").search(game.get_initial_state())
    assert abs(float(policy.sum()) - 1.0) < 1e-5
