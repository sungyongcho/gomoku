import multiprocessing as mp
from types import SimpleNamespace

import numpy as np
import pytest
import ray

from gomoku.alphazero.runners.selfplay import SelfPlayRunner
from gomoku.alphazero.runners.vectorize_runner import VectorizeRunner
from gomoku.alphazero.agent import AlphaZeroAgent
from gomoku.core.gomoku import Gomoku
from gomoku.inference.local import LocalInference
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import RootConfig, load_and_parse_config


@pytest.fixture(scope="function")
def sequential_components() -> tuple[
    RootConfig, Gomoku, AlphaZeroAgent, SelfPlayRunner
]:
    """Phase 1(Sequential) 테스트용 통합 구성."""
    cfg_path = "configs/config_alphazero_test.yaml"
    cfg = load_and_parse_config(cfg_path)
    game = Gomoku(cfg.board)
    model = PolicyValueNet(game, cfg.model, device="cpu")
    inference = LocalInference(model)
    agent = AlphaZeroAgent(
        game=game,
        mcts_cfg=cfg.mcts,
        inference_client=inference,
        engine_type="sequential",
    )
    selfplay = SelfPlayRunner(game=game, mcts_cfg=cfg.mcts, train_cfg=cfg.training)
    return cfg, game, agent, selfplay


@pytest.fixture(scope="function")
def vectorize_components() -> tuple[
    RootConfig, Gomoku, AlphaZeroAgent, VectorizeRunner, LocalInference
]:
    """Phase 2(Vectorize) 테스트용 통합 구성."""
    cfg_path = "configs/config_alphazero_vectorize_test.yaml"
    cfg = load_and_parse_config(cfg_path)
    game = Gomoku(cfg.board)
    model = PolicyValueNet(game, cfg.model, device="cpu")
    inference = LocalInference(model)
    agent = AlphaZeroAgent(
        game=game,
        mcts_cfg=cfg.mcts,
        inference_client=inference,
        engine_type="vectorize",
    )
    runner = VectorizeRunner(game=game, mcts_cfg=cfg.mcts, train_cfg=cfg.training)
    return cfg, game, agent, runner, inference


@pytest.fixture(scope="function")
def mp_components() -> tuple[RootConfig, mp.Queue, list[mp.Queue], np.ndarray]:
    """MP 서버/워커 테스트용 큐와 설정을 준비한다."""
    cfg_path = "configs/config_alphazero_mp_test.yaml"
    cfg = load_and_parse_config(cfg_path)
    ctx = mp.get_context("spawn")
    request_q: mp.Queue = ctx.Queue()
    result_queues: list[mp.Queue] = [ctx.Queue() for _ in range(2)]

    sample_state = np.ones(
        (cfg.model.num_planes, cfg.board.num_lines, cfg.board.num_lines),
        dtype=np.float32,
    )
    return cfg, request_q, result_queues, sample_state


@pytest.fixture(scope="function")
def ray_local_ctx() -> None:
    """테스트 전후로 Ray를 안전하게 초기화/종료한다."""
    if ray.is_initialized():
        ray.shutdown()
    # local_mode는 deprecated이므로 실제 클러스터 모드에서 자원만 최소화한다.
    ray.init(num_cpus=2, ignore_reinit_error=True, include_dashboard=False)
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope="function")
def dummy_record() -> tuple:
    """GameRecord 더미와 게임 객체를 반환한다."""
    cfg = load_and_parse_config("configs/config_alphazero_vectorize_test.yaml")
    game = Gomoku(cfg.board)
    s0 = game.get_initial_state()
    x, y = 0, 0
    s1 = game.get_next_state(s0, (x, y), player=1)
    pi0 = np.zeros(game.action_size, dtype=np.float32)
    pi0[0] = 1.0
    pi1 = np.zeros(game.action_size, dtype=np.float32)
    pi1[1 if game.action_size > 1 else 0] = 1.0
    moves = np.asarray([0, 1], dtype=np.int32)
    players = np.asarray([1, 2], dtype=np.int8)
    outcomes = np.asarray([1, -1], dtype=np.int8)
    rec = SimpleNamespace(
        states_raw=[s0, s1],
        policies=np.stack([pi0, pi1]),
        moves=moves,
        players=players,
        outcomes=outcomes,
        config_snapshot={"test": True},
        priority=1.0,
    )
    return game, rec
