"""System integration checks for PolicyValueNet."""

import io

import pytest
import torch
from torch.testing import assert_close

from gomoku.core.gomoku import Gomoku
from gomoku.model.model_helpers import calc_num_hidden, calc_num_resblocks
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import ModelConfig
from tests.helpers import make_game, make_model


def _run_forward(model: PolicyValueNet, game: Gomoku, batch: int = 1) -> PolicyValueNet:
    """Run a forward pass with random input matching model channels."""
    in_ch = model.start_block[0].in_channels
    x = torch.randn(batch, in_ch, game.row_count, game.col_count)
    with torch.no_grad():
        return model(x)


def test_variable_board_size_end_to_end():
    """Encoding→forward should work for multiple board sizes."""
    for size in (9, 15, 19):
        game = make_game(num_lines=size)
        num_planes = int(game.get_encoded_state(game.get_initial_state()).shape[1])
        config = ModelConfig(
            num_planes=num_planes,
            num_hidden=calc_num_hidden(size),
            num_resblocks=calc_num_resblocks(size),
            policy_channels=2,
            value_channels=1,
        )
        model = PolicyValueNet(game, config, device="cpu")
        policy, value = _run_forward(model, game, batch=2)
        assert policy.shape == (2, game.action_size)
        assert value.shape == (2, 1)


def test_batch_processing_consistency():
    """Batch forward should match single forward on same sample."""
    model, game, config = make_model()
    model.eval()
    x = torch.randn(3, config.num_planes, game.row_count, game.col_count)
    with torch.no_grad():
        batch_policy, batch_value = model(x)
        single_policy, single_value = model(x[:1])

    diff_policy = (batch_policy[0] - single_policy[0]).abs().max()
    diff_value = (batch_value[0] - single_value[0]).abs().max()
    print(f"Max Policy Diff: {diff_policy:.2e}")
    print(f"Max Value Diff: {diff_value:.2e}")

    assert torch.allclose(batch_policy[0], single_policy[0], atol=1e-5), (
        f"Policy output mismatch. Max diff: {diff_policy}"
    )

    assert torch.allclose(batch_value[0], single_value[0], atol=1e-5), (
        f"Value output mismatch. Max diff: {diff_value}"
    )


def test_serialization_and_device_mobility():
    """state_dict save/load and CPU↔CPU roundtrip should preserve outputs."""
    model, game, config = make_model()
    model.eval()
    x = torch.randn(2, config.num_planes, game.row_count, game.col_count)
    with torch.no_grad():
        orig_policy, orig_value = model(x)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    loaded_model, _, _ = make_model()
    loaded_model.load_state_dict(torch.load(buffer, map_location="cpu"))
    loaded_model.eval()
    with torch.no_grad():
        loaded_policy, loaded_value = loaded_model(x)
    assert_close(orig_policy, loaded_policy)
    assert_close(orig_value, loaded_value)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA compatible GPU"
)
def test_device_mobility_and_consistency():
    """Model and inputs should move to CUDA and keep outputs consistent."""
    model, game, config = make_model()
    model.eval()
    x_cpu = torch.randn(2, config.num_planes, game.row_count, game.col_count)
    with torch.no_grad():
        cpu_policy, cpu_value = model(x_cpu)

    device = torch.device("cuda")
    model.to(device)
    x_gpu = x_cpu.to(device)

    first_param = next(model.parameters())
    assert first_param.device.type == "cuda"
    assert x_gpu.device.type == "cuda"

    with torch.no_grad():
        gpu_policy, gpu_value = model(x_gpu)

    assert torch.allclose(cpu_policy, gpu_policy.cpu(), atol=1e-4)
    assert torch.allclose(cpu_value, gpu_value.cpu(), atol=1e-4)

    model.cpu()
    with torch.no_grad():
        back_policy, back_value = model(x_cpu)
    assert torch.allclose(cpu_policy, back_policy, atol=1e-5)
    assert torch.allclose(cpu_value, back_value, atol=1e-5)


def test_mismatched_input_channels_raises_error():
    """Mismatched input channels should raise a clear error."""
    model, game, config = make_model()
    bad = torch.randn(1, config.num_planes + 1, game.row_count, game.col_count)
    try:
        model(bad)
    except RuntimeError as exc:  # expected torch shape error
        assert (
            "expected input" in str(exc).lower() or "size mismatch" in str(exc).lower()
        )
    else:
        raise AssertionError("Expected RuntimeError for mismatched channels")


def test_seed_reproducibility_for_model_init_and_forward():
    """With fixed seed, init+forward should be reproducible."""
    torch.manual_seed(123)
    model1, game, config = make_model()
    x = torch.randn(2, config.num_planes, game.row_count, game.col_count)
    with torch.no_grad():
        p1, v1 = model1(x)

    torch.manual_seed(123)
    model2, _, _ = make_model()
    with torch.no_grad():
        p2, v2 = model2(x)

    assert_close(p1, p2)
    assert_close(v1, v2)
