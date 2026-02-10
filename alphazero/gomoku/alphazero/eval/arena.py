from gomoku.alphazero.agent import AlphaZeroAgent
from gomoku.alphazero.eval.match import play_match
from gomoku.alphazero.eval.metrics import H2HMetrics
from gomoku.alphazero.eval.sprt import check_sprt
from gomoku.core.gomoku import Gomoku
from gomoku.inference.base import InferenceClient
from gomoku.pvmcts.pvmcts import PVMCTS
from gomoku.utils.config.loader import (
    EvaluationConfig,
    MctsConfig,
    RootConfig,
    SPRTConfig,
)
from gomoku.utils.config.schedule_param import get_scheduled_value
from gomoku.utils.progress import make_progress


def _build_eval_mcts(
    cfg: RootConfig,
    inference: InferenceClient,
    eval_cfg: EvaluationConfig,
    async_inflight_limit: int | None = None,
) -> PVMCTS:
    """Build an evaluation-only PVMCTS instance.

    Parameters
    ----------
    cfg : RootConfig
        Root configuration containing board and base MCTS settings.
    inference : InferenceClient
        Inference backend for value/policy predictions.
    eval_cfg : EvaluationConfig
        Evaluation overrides for searches and noise.
    async_inflight_limit : int | None, optional
        Limit for async/pipelined requests if engine is ray.

    Returns
    -------
    PVMCTS
        Configured MCTS instance for evaluation (noise disabled).
    """
    game = Gomoku(cfg.board)
    mcts_cfg = cfg.mcts
    # Enforce evaluation settings: disable noise/temperature, optionally raise searches
    eval_searches = int(
        get_scheduled_value(
            getattr(eval_cfg, "eval_num_searches", mcts_cfg.num_searches),
            0,
        )
    )
    eval_dir_eps = float(getattr(eval_cfg, "eval_dirichlet_epsilon", 0.0))
    eval_mcts_cfg = MctsConfig(
        C=mcts_cfg.C,
        num_searches=eval_searches,
        exploration_turns=int(get_scheduled_value(mcts_cfg.exploration_turns, 0)),
        dirichlet_epsilon=eval_dir_eps,
        dirichlet_alpha=float(get_scheduled_value(mcts_cfg.dirichlet_alpha, 0)),
        batch_infer_size=mcts_cfg.batch_infer_size,
        max_batch_wait_ms=mcts_cfg.max_batch_wait_ms,
        min_batch_size=mcts_cfg.min_batch_size,
    )
    # Auto-detect engine type: use "ray" if client is async, else "sequential"
    engine_type = "sequential"
    if hasattr(inference, "infer_async"):
        # Very simplistic check: RayInferenceClient has proper async
        engine_type = "ray"

    agent = AlphaZeroAgent(
        game=game,
        mcts_cfg=eval_mcts_cfg,
        inference_client=inference,
        engine_type=engine_type,
        async_inflight_limit=async_inflight_limit,
    )
    return agent.mcts


def _run_duel(
    cfg: RootConfig,
    eval_cfg: EvaluationConfig,
    inference_a: InferenceClient,
    inference_b: InferenceClient,
    games: int,
    *,
    opening_turns: int,
    temperature: float,
    blunder_th: float,
    progress_desc: str | None = None,
    first_player: str | None = None,
    metrics: H2HMetrics | None = None,
    sprt_state: dict[str, int] | None = None,
    async_inflight_limit: int | None = None,
) -> H2HMetrics:
    """Run head-to-head games with alternating colors under identical settings.

    Parameters
    ----------
    cfg : RootConfig
        Root configuration with board/MCTS settings.
    eval_cfg : EvaluationConfig
        Evaluation configuration including SPRT options.
    inference_a : InferenceClient
        Challenger inference backend.
    inference_b : InferenceClient
        Opponent inference backend.
    games : int
        Maximum number of games to play.
    opening_turns : int
        Turns to sample before switching to argmax.
    temperature : float
        Sampling temperature for the opening phase.
    blunder_th : float
        Q-drop threshold for counting blunders.
    progress_desc : str | None
        tqdm description to display.
    first_player : str | None
        Force the starting player for all games. Use ``"a"`` for inference_a as
        Player 1, ``"b"`` for inference_b as Player 1, or ``None`` to alternate.
    metrics : H2HMetrics | None
        Optional metrics container to accumulate into; when omitted, a fresh
        container is created.
    sprt_state : dict[str, int] | None
        Mutable SPRT state to carry across multiple duels. When ``None``, a new
        state is initialized.

    Returns
    -------
    H2HMetrics
        Aggregated head-to-head statistics.
    """
    game = Gomoku(cfg.board)
    metrics = metrics or H2HMetrics()
    sprt_cfg: SPRTConfig | None = (
        getattr(eval_cfg, "sprt", None)
        if getattr(eval_cfg, "use_sprt", False)
        else None
    )
    sprt_state = sprt_state or {"wins": 0, "losses": 0, "draws": 0}
    progress = make_progress(
        total=games, desc=progress_desc, unit="game", disable=progress_desc is None
    )
    for i in range(games):
        mcts_a = _build_eval_mcts(cfg, inference_a, eval_cfg, async_inflight_limit)
        mcts_b = _build_eval_mcts(cfg, inference_b, eval_cfg, async_inflight_limit)

        use_a_first = True
        flip_sign = False
        if first_player == "a":
            use_a_first = True
        elif first_player == "b":
            use_a_first = False
            flip_sign = True
        else:
            use_a_first = i % 2 == 0
            flip_sign = not use_a_first

        if use_a_first:
            winner, blunders, moves = play_match(
                game,
                mcts_a,
                mcts_b,
                opening_turns=opening_turns,
                temperature=temperature,
                blunder_threshold=blunder_th,
            )
        else:
            winner, blunders, moves = play_match(
                game,
                mcts_b,
                mcts_a,
                opening_turns=opening_turns,
                temperature=temperature,
                blunder_threshold=blunder_th,
            )
            flip_sign = True

        if flip_sign:
            winner = -winner  # flip to inference_a perspective

        metrics.record_game(winner, blunders, moves)
        progress.update(1)
        if sprt_cfg is not None:
            if winner > 0:
                sprt_state["wins"] += 1
            elif winner < 0:
                sprt_state["losses"] += 1
            else:
                sprt_state["draws"] += 1
            decision = check_sprt(
                sprt_cfg, sprt_state["wins"], sprt_state["losses"], sprt_state["draws"]
            )
            if decision in {"accept_h1", "accept_h0"}:
                break
    progress.close()
    return metrics


def run_arena(
    cfg: RootConfig,
    inference_new: InferenceClient,
    inference_best: InferenceClient,
    *,
    eval_cfg: EvaluationConfig,
    matches: int,
    baseline_inference: InferenceClient | None = None,
) -> dict[str, float]:
    """Evaluate a challenger against the champion (and optional baseline).

    Parameters
    ----------
    cfg : RootConfig
        Root configuration for the game and MCTS.
    inference_new : InferenceClient
        Inference backend for the challenger.
    inference_best : InferenceClient
        Inference backend for the current champion.
    eval_cfg : EvaluationConfig
        Evaluation settings including promotion thresholds.
    matches : int
        Number of head-to-head games to attempt.
    baseline_inference : InferenceClient | None
        Optional baseline model to compare against.

    Returns
    -------
    dict[str, float]
        Summary metrics including promotion decision flags.
    """
    opening_turns = int(getattr(eval_cfg, "eval_opening_turns", 0))
    temp = float(getattr(eval_cfg, "eval_temperature", 0.0))
    blunder_th = float(getattr(eval_cfg, "blunder_threshold", 0.0))

    p1_games = (matches + 1) // 2
    p2_games = matches // 2

    arena_label_p1 = "Arena | P1: Challenger | P2: Champion"
    arena_label_p2 = "Arena | P1: Champion | P2: Challenger"
    h2h_metrics = H2HMetrics()
    sprt_state = (
        {"wins": 0, "losses": 0, "draws": 0}
        if getattr(eval_cfg, "use_sprt", False)
        else None
    )

    if p1_games > 0:
        ret_metrics = _run_duel(
            cfg,
            eval_cfg,
            inference_new,
            inference_best,
            p1_games,
            opening_turns=opening_turns,
            temperature=temp,
            blunder_th=blunder_th,
            progress_desc=arena_label_p1,
            first_player="a",
            metrics=h2h_metrics,
            sprt_state=sprt_state,
        )
        if isinstance(ret_metrics, H2HMetrics):
            h2h_metrics = ret_metrics

    games_played = h2h_metrics.total_games
    if p2_games > 0 and games_played >= p1_games:
        ret_metrics = _run_duel(
            cfg,
            eval_cfg,
            inference_new,
            inference_best,
            p2_games,
            opening_turns=opening_turns,
            temperature=temp,
            blunder_th=blunder_th,
            progress_desc=arena_label_p2,
            first_player="b",
            metrics=h2h_metrics,
            sprt_state=sprt_state,
        )
        if isinstance(ret_metrics, H2HMetrics):
            h2h_metrics = ret_metrics

    summary: dict[str, float] = h2h_metrics.summary()
    summary["promotion_win_rate"] = float(
        get_scheduled_value(getattr(eval_cfg, "promotion_win_rate", 0.0), 0)
    )

    if baseline_inference is not None and eval_cfg.num_baseline_games > 0:
        baseline_games = int(eval_cfg.num_baseline_games)
        base_p1_games = (baseline_games + 1) // 2
        base_p2_games = baseline_games // 2
        baseline_label_p1 = "Baseline | P1: Challenger | P2: Baseline"
        baseline_label_p2 = "Baseline | P1: Baseline | P2: Challenger"
        base_metrics = H2HMetrics()

        if base_p1_games > 0:
            try:
                ret_base = _run_duel(
                    cfg,
                    eval_cfg,
                    inference_new,
                    baseline_inference,
                    base_p1_games,
                    opening_turns=opening_turns,
                    temperature=temp,
                    blunder_th=blunder_th,
                    progress_desc=baseline_label_p1,
                    first_player="a",
                    metrics=base_metrics,
                )
                if isinstance(ret_base, H2HMetrics):
                    base_metrics = ret_base
            except Exception as exc:  # noqa: BLE001
                print(f"[Arena] Baseline P1 duel skipped due to error: {exc}")

        base_games_played = base_metrics.total_games
        if base_p2_games > 0 and base_games_played >= base_p1_games:
            try:
                ret_base = _run_duel(
                    cfg,
                    eval_cfg,
                    inference_new,
                    baseline_inference,
                    base_p2_games,
                    opening_turns=opening_turns,
                    temperature=temp,
                    blunder_th=blunder_th,
                    progress_desc=baseline_label_p2,
                    first_player="b",
                    metrics=base_metrics,
                )
                if isinstance(ret_base, H2HMetrics):
                    base_metrics = ret_base
            except Exception as exc:  # noqa: BLE001
                print(f"[Arena] Baseline P2 duel skipped due to error: {exc}")

        base_summary = base_metrics.summary()
        if not base_summary:
            print(
                "[Arena] Baseline summary empty; falling back to challenger vs champion stats."
            )
            base_summary = summary
        summary["baseline_win_rate"] = base_summary.get("win_rate", 0.0)
        summary["baseline_blunder_rate"] = base_summary.get("blunder_rate", 0.0)

    # Evaluate guardrails/promotion conditions
    target_wr = summary.get("win_rate")
    baseline_wr = summary.get("baseline_win_rate")
    blunder_rate = summary.get("blunder_rate")

    promote_wr = summary["promotion_win_rate"]
    baseline_min = eval_cfg.baseline_wr_min
    blunder_limit = eval_cfg.blunder_increase_limit

    summary["baseline_win_rate_required"] = float(baseline_min)
    summary["blunder_rate_limit"] = float(blunder_limit)
    summary["pass_promotion"] = bool(target_wr >= promote_wr)
    if "baseline_win_rate" in summary:
        summary["pass_baseline"] = bool(baseline_wr >= baseline_min)
    else:
        summary["pass_baseline"] = True
    summary["pass_blunder"] = bool(blunder_rate <= blunder_limit)

    summary["promote"] = bool(
        summary["pass_promotion"]
        and summary["pass_baseline"]
        and summary["pass_blunder"]
    )

    return summary
