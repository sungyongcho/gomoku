import torch

from gomoku.pvmcts.treenode import TreeNode
from gomoku.utils.config.loader import MctsConfig


def test_vectorize_native_smoke(native_game) -> None:
    from gomoku.pvmcts.search.vectorize import VectorizeEngine

    cfg = MctsConfig(
        num_searches=2,
        C=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        exploration_turns=0,
        batch_infer_size=1,
        max_batch_wait_ms=0,
        min_batch_size=1,
    )

    class DummyInference:
        def infer_batch(self, batch):
            b = batch.shape[0]
            logits = torch.zeros((b, native_game.action_size), dtype=torch.float32)
            values = torch.ones((b,), dtype=torch.float32)
            return logits, values

    engine = VectorizeEngine(native_game, cfg, DummyInference(), use_fp16=False)
    root = TreeNode(native_game.get_initial_state(), None, None, 1)
    engine.search(root, add_noise=False)
    print("vectorize native visit_count:", root.visit_count)
    assert root.visit_count >= 1


def test_sequential_native_smoke(native_game) -> None:
    from gomoku.pvmcts.search.sequential import SequentialEngine

    cfg = MctsConfig(
        num_searches=1,
        C=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,
        exploration_turns=0,
        batch_infer_size=1,
        max_batch_wait_ms=0,
        min_batch_size=1,
    )

    class DummyInference:
        def infer(self, batch):
            logits = torch.zeros((native_game.action_size,), dtype=torch.float32)
            value = torch.tensor([1.0], dtype=torch.float32)
            return logits, value

    engine = SequentialEngine(native_game, cfg, DummyInference(), use_fp16=False)
    root = TreeNode(native_game.get_initial_state(), None, None, 1)
    engine.search(root, add_noise=False)
    print("sequential native visit_count:", root.visit_count)
    assert root.visit_count >= 1


def test_mp_native_smoke(native_game) -> None:
    from gomoku.pvmcts.search.mp import MultiprocessEngine

    cfg = MctsConfig(
        num_searches=1,
        C=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,
        exploration_turns=0,
        batch_infer_size=1,
        max_batch_wait_ms=0,
        min_batch_size=1,
    )

    class DummyMPClient:
        def __init__(self):
            self.device = torch.device("cpu")

        def infer(self, batch):
            b = batch.shape[0]
            logits = torch.zeros((b, native_game.action_size), dtype=torch.float32)
            values = torch.ones((b,), dtype=torch.float32)
            return logits, values

    engine = MultiprocessEngine(
        native_game,
        cfg,
        inference=DummyMPClient(),
        batch_size=1,
        use_fp16=False,
    )
    root = TreeNode(native_game.get_initial_state(), None, None, 1)
    engine.search(root, add_noise=False)
    print("mp native visit_count:", root.visit_count)
    assert root.visit_count >= 1


def test_ray_native_smoke(monkeypatch, native_game) -> None:
    import gomoku.pvmcts.search.ray.batch_inference_manager as bim
    import gomoku.pvmcts.search.ray.ray_async as ray_async_mod
    from gomoku.pvmcts.search.ray.ray_async import RayAsyncEngine

    class DummyRayClient:
        device = torch.device("cpu")

        def __init__(self):
            self.calls = []

        def infer(self, states, native_payload=None):
            b = states.shape[0] if states.dim() == 4 else 1
            logits = torch.zeros((b, native_game.action_size), dtype=torch.float32)
            values = torch.ones((b,), dtype=torch.float32)
            self.calls.append((states.shape, native_payload))
            return logits, values

        def infer_async(self, states, native_payload=None):
            logits, values = self.infer(states, native_payload)

            class _Ref:
                def __init__(self, p, v):
                    self.p = p
                    self.v = v

            return _Ref(logits, values)

    def _dummy_wait(refs, num_returns, timeout, fetch_local):
        return refs, []

    class DummyBatchResult:
        def __init__(self, mapping, policy_logits, values):
            self.mapping = mapping
            self.policy_logits = policy_logits
            self.values = values

    class DummyManager:
        def __init__(
            self, client, batch_size, max_wait_ms=None, max_inflight_batches=None
        ):
            self.client = client
            self.queue = []

        def enqueue(self, mapping, tensor, native_state=None):
            self.queue.append((mapping, tensor, native_state))

        def dispatch_ready(self, force: bool = False):
            if not self.queue:
                return False
            mapping, tensor, native_state = self.queue.pop(0)
            logits, values = self.client.infer(tensor, native_payload=[native_state])
            self.result = DummyBatchResult([mapping], logits, values)
            return False

        def drain_ready(self, timeout_s: float = 0.0):
            return [self.result] if hasattr(self, "result") else []

        def pending_count(self):
            return len(self.queue)

        def cleanup(self):
            self.queue.clear()

    monkeypatch.setattr(bim, "BatchInferenceManager", DummyManager)
    monkeypatch.setattr(ray_async_mod, "BatchInferenceManager", DummyManager)
    monkeypatch.setattr("ray.wait", _dummy_wait, raising=False)

    cfg = MctsConfig(
        num_searches=1,
        C=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,
        exploration_turns=0,
        batch_infer_size=1,
        max_batch_wait_ms=0,
        min_batch_size=1,
    )
    client = DummyRayClient()
    engine = RayAsyncEngine(
        native_game,
        cfg,
        client,
        batch_size=1,
        async_inflight_limit=1,
        use_fp16=False,
    )
    root = TreeNode(native_game.get_initial_state(), None, None, 1)
    engine.search(root, add_noise=False)
    print("ray native visit_count:", root.visit_count)
    assert root.visit_count >= 1


def test_mp_inference_broken_pipe_fallback(monkeypatch, native_game) -> None:
    from gomoku.pvmcts.search.mp import MultiprocessEngine

    class FailingMPClient:
        device = torch.device("cpu")

        def infer(self, batch):
            raise BrokenPipeError("simulated broken pipe")

    cfg = MctsConfig(
        num_searches=1,
        C=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,
        exploration_turns=0,
        batch_infer_size=1,
        max_batch_wait_ms=0,
        min_batch_size=1,
    )

    engine = MultiprocessEngine(
        native_game,
        cfg,
        inference=FailingMPClient(),
        batch_size=1,
        use_fp16=False,
    )
    root = TreeNode(native_game.get_initial_state(), None, None, 1)
    engine.search(root, add_noise=False)
    print("broken pipe fallback visit_count:", root.visit_count)
    assert root.visit_count in (0, 1)
