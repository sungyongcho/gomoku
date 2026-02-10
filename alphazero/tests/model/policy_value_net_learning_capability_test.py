"""Learning capability checks for PolicyValueNet."""

import torch
from torch import nn

from gomoku.core.game_config import PLAYER_1
from tests.helpers import log_section, make_model


def test_single_batch_overfit():
    """Model should overfit a tiny batch (loss decreases substantially)."""
    log_section("PolicyValueNet - Single batch overfit")
    torch.manual_seed(7)
    model, game, _ = make_model()
    model.train()

    state = game.get_initial_state()
    state = game.get_next_state(state, (2, 2), PLAYER_1)
    encoded = torch.tensor(game.get_encoded_state(state), dtype=torch.float32)
    batch_size = 8
    x = encoded.repeat(batch_size, 1, 1, 1)

    target_move = 2 * game.col_count + 2
    target_policy = torch.full((batch_size,), target_move, dtype=torch.long)
    target_value = torch.ones((batch_size, 1), dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    with torch.no_grad():
        init_policy, init_value = model(x)
        initial_loss = ce(init_policy, target_policy) + mse(init_value, target_value)

    final_loss = None
    for _ in range(400):
        optimizer.zero_grad()
        policy_logits, value = model(x)
        loss = ce(policy_logits, target_policy) + mse(value, target_value)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    print("Initial loss:", float(initial_loss))
    print("Final loss:", final_loss)

    assert final_loss is not None
    assert final_loss < float(initial_loss) * 0.2
    with torch.no_grad():
        policy_logits, value = model(x)
    preds = policy_logits.argmax(dim=1)
    assert torch.all(preds == target_policy)
    print("Final predicted value:", value.mean().item())
    assert torch.allclose(value, target_value, atol=0.05)
