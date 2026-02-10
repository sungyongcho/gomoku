import time
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator
import yaml

from gomoku.model.model_helpers import (
    POLICY_CHANNELS,
    VALUE_CHANNELS,
    calc_num_hidden,
    calc_num_planes,
    calc_num_resblocks,
)
from gomoku.utils.config.schedule_param import (
    ScheduledFloat,
    parse_scheduled_float,
    validate_against_num_iterations,
)

# Path invariants: these are fixed; only run_id and GCS toggle/bucket may vary.
PATH_RUN_PREFIX = "runs"
PATH_REPLAY_DIR = "{run_prefix}/{run_id}/replay"
PATH_CKPT_DIR = "{run_prefix}/{run_id}/ckpt"
PATH_EVAL_LOGS_DIR = "{run_prefix}/{run_id}/eval_logs"
PATH_MANIFEST = "{run_prefix}/{run_id}/manifest.json"


class BaseConfig(BaseModel):
    """Common base for config models (strict & frozen)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class BoardConfig(BaseConfig):
    num_lines: int
    enable_doublethree: bool
    enable_capture: bool
    capture_goal: int
    gomoku_goal: int
    history_length: int = 5


class ModelConfig(BaseConfig):
    num_planes: int | None = None
    num_hidden: int | None = None
    num_resblocks: int | None = None
    policy_channels: int = POLICY_CHANNELS
    value_channels: int = VALUE_CHANNELS


class PriorityReplayConfig(BaseConfig):
    enabled: bool = False
    start_iteration: int = 0
    alpha: float = 0.6
    beta: float = 0.4
    epsilon: float = 1e-3
    trigger_no_promotion_iters: int | None = None


class OpponentConfig(BaseConfig):
    random_bot_ratio: ScheduledFloat = 0.0
    prev_bot_ratio: ScheduledFloat = 0.0
    past_model_window: int = 5


class TrainingConfig(BaseConfig):
    num_iterations: int
    num_selfplay_iterations: ScheduledFloat
    num_epochs: int
    batch_size: int
    learning_rate: ScheduledFloat
    weight_decay: float
    temperature: ScheduledFloat
    replay_buffer_size: int
    min_samples_to_train: int
    # -- Opening / exploration randomness --
    # random_play_ratio: 매 턴 독립적으로 확률 p만큼 uniform random 수를 둔다.
    #   게임 전체에 적용되므로 중반·종반 품질도 낮아짐. 작은 값(0.01~0.03) 권장.
    # random_opening_turns: 게임 시작 후 처음 N수를 무조건 uniform random으로 둔다.
    #   opening 다양성만 높이고, N수 이후에는 정상 MCTS를 사용하므로
    #   중·종반 품질에 영향 없음. opening 고착 해소에 적합.
    random_play_ratio: ScheduledFloat | None = None
    random_opening_turns: ScheduledFloat | None = None
    opponent_rates: OpponentConfig | None = None
    priority_replay: PriorityReplayConfig | None = None
    dataloader_num_workers: int = 1
    dataloader_prefetch_factor: int = 1
    enable_tf32: bool = True
    use_channels_last: bool = False


class MctsConfig(BaseConfig):
    C: float
    num_searches: ScheduledFloat
    exploration_turns: ScheduledFloat
    dirichlet_epsilon: ScheduledFloat
    dirichlet_alpha: ScheduledFloat
    batch_infer_size: int
    max_batch_wait_ms: int
    min_batch_size: int
    use_native: bool = False
    # Adjudication (Early Resignation) Settings
    resign_threshold: float = 0.95
    resign_enabled: bool = True
    min_moves_before_resign: int = 10


class SPRTConfig(BaseConfig):
    p0: float
    p1: float
    alpha: float
    beta: float
    max_games: int
    ignore_draws: bool
    soft_margin_ratio: float | None = None
    soft_accept_threshold: float | None = None
    soft_reject_threshold: float | None = None
    fallback_after_games: int | None = None
    fallback_accept_threshold: float | None = None
    fallback_reject_threshold: float | None = None


class FastEvalConfig(BaseConfig):
    enabled: bool = False
    num_games: int = 0
    num_searches: int = 0
    promote_threshold: float = 0.0
    reject_threshold: float = 0.0


class EvaluationConfig(BaseConfig):
    num_eval_games: int
    eval_every_iters: int
    promotion_win_rate: ScheduledFloat
    num_baseline_games: int
    blunder_threshold: float
    initial_blunder_rate: float
    initial_baseline_win_rate: float
    blunder_increase_limit: float
    baseline_wr_min: float
    random_play_ratio: ScheduledFloat
    eval_opening_turns: int | None = None
    eval_temperature: float | None = None
    eval_dirichlet_epsilon: float | None = None
    eval_num_searches: ScheduledFloat | None = None
    baseline_num_searches: ScheduledFloat | None = None
    use_sprt: bool = False
    sprt: SPRTConfig | None = None
    fast_eval: FastEvalConfig | None = None
    adjudication_win_prob: float | None = None
    adjudication_min_turns: int | None = None
    elo_k_factor: float = 2.0


class ParallelConfig(BaseConfig):
    num_parallel_games: int | None = None
    mp_num_workers: int | None = None
    ray_local_num_workers: int | None = None


class PathsConfig(BaseConfig):
    use_gcs: bool = False
    run_prefix: str | None = None
    run_id: str | None = None
    replay_dir: str = PATH_REPLAY_DIR
    ckpt_dir: str = PATH_CKPT_DIR
    evaluation_logs_dir: str = PATH_EVAL_LOGS_DIR
    manifest: str = PATH_MANIFEST


class IoConfig(BaseConfig):
    initial_replay_shards: int | None = None
    initial_replay_iters: int | None = None
    max_samples_per_shard: int | None = None
    local_replay_cache: str | None = None


class SelfPlayRuntimeConfig(BaseConfig):
    min_floor: int | None = None
    timeout_s: int | None = None
    util_factor: float | None = None
    actor_num_cpus: float | None = 1.0
    games_per_actor: int | None = None
    inflight_per_actor: int | None = None
    random_num_searches: int | None = None
    max_chunk_size: int | None = None
    random_max_chunk_size: int | None = None


class InferenceRuntimeConfig(BaseConfig):
    keep_on_gpu: bool = False
    warmup_batches: int = 0
    actor_num_gpus: float | None = None
    num_actors: int | None = None
    actor_num_cpus: float | None = 1.0
    num_servers: int | None = None
    use_local_inference: bool = False


class EvaluationRuntimeConfig(BaseConfig):
    num_workers: int | None = None
    actor_num_cpus: float | None = None
    actor_num_gpus: float | None = None
    max_games_per_worker: int | None = None
    scheduling_strategy: str | None = None
    wait_timeout_s: float | None = None
    stall_timeout_s: float | None = None
    debug_eval_events: bool = False


class RuntimeConfig(BaseConfig):
    selfplay: SelfPlayRuntimeConfig = SelfPlayRuntimeConfig()
    inference: InferenceRuntimeConfig = InferenceRuntimeConfig()
    evaluation: EvaluationRuntimeConfig = EvaluationRuntimeConfig()


class RootConfig(BaseConfig):
    board: BoardConfig
    model: ModelConfig
    training: TrainingConfig
    mcts: MctsConfig
    evaluation: EvaluationConfig
    parallel: ParallelConfig
    paths: PathsConfig
    io: IoConfig
    runtime: RuntimeConfig | None = None

    @model_validator(mode="after")
    def _post_init(self) -> "RootConfig":
        num_iterations = int(self.training.num_iterations)

        def _schedule(value: Any, name: str) -> ScheduledFloat:
            return parse_scheduled_float(
                value,
                name=name,
                num_iterations=num_iterations,
            )

        expected_planes = calc_num_planes(self.board.history_length)
        if self.model.num_planes not in (None, expected_planes):
            raise ValueError(
                f"model.num_planes is fixed to {expected_planes}; remove overrides in YAML."
            )
        if self.model.policy_channels != POLICY_CHANNELS:
            raise ValueError(
                f"model.policy_channels is fixed to {POLICY_CHANNELS}; remove overrides in YAML."
            )
        if self.model.value_channels != VALUE_CHANNELS:
            raise ValueError(
                f"model.value_channels is fixed to {VALUE_CHANNELS}; remove overrides in YAML."
            )

        model_updates: dict[str, Any] = {
            "num_planes": expected_planes,
            "policy_channels": POLICY_CHANNELS,
            "value_channels": VALUE_CHANNELS,
        }
        if self.model.num_hidden is None:
            model_updates["num_hidden"] = calc_num_hidden(self.board.num_lines)
        if self.model.num_resblocks is None:
            model_updates["num_resblocks"] = calc_num_resblocks(self.board.num_lines)

        model = self.model.model_copy(update=model_updates)

        opponent_rates = (
            self.training.opponent_rates.model_copy(
                update={
                    "random_bot_ratio": _schedule(
                        self.training.opponent_rates.random_bot_ratio,
                        "training.opponent_rates.random_bot_ratio",
                    ),
                    "prev_bot_ratio": _schedule(
                        self.training.opponent_rates.prev_bot_ratio,
                        "training.opponent_rates.prev_bot_ratio",
                    ),
                }
            )
            if self.training.opponent_rates is not None
            else None
        )

        training = self.training.model_copy(
            update={
                "num_selfplay_iterations": _schedule(
                    self.training.num_selfplay_iterations,
                    "training.num_selfplay_iterations",
                ),
                "learning_rate": _schedule(
                    self.training.learning_rate, "training.learning_rate"
                ),
                "temperature": _schedule(
                    self.training.temperature, "training.temperature"
                ),
                "random_play_ratio": _schedule(
                    self.training.random_play_ratio, "training.random_play_ratio"
                )
                if self.training.random_play_ratio is not None
                else None,
                "opponent_rates": opponent_rates,
            }
        )

        mcts = self.mcts.model_copy(
            update={
                "num_searches": _schedule(self.mcts.num_searches, "mcts.num_searches"),
                "dirichlet_epsilon": _schedule(
                    self.mcts.dirichlet_epsilon, "mcts.dirichlet_epsilon"
                ),
            }
        )

        evaluation = self.evaluation.model_copy(
            update={
                "promotion_win_rate": _schedule(
                    self.evaluation.promotion_win_rate,
                    "evaluation.promotion_win_rate",
                ),
                "random_play_ratio": _schedule(
                    self.evaluation.random_play_ratio, "evaluation.random_play_ratio"
                ),
                "eval_num_searches": _schedule(
                    self.evaluation.eval_num_searches,
                    "evaluation.eval_num_searches",
                )
                if self.evaluation.eval_num_searches is not None
                else None,
                "baseline_num_searches": _schedule(
                    self.evaluation.baseline_num_searches,
                    "evaluation.baseline_num_searches",
                )
                if self.evaluation.baseline_num_searches is not None
                else None,
            }
        )

        schedules_to_validate = {
            "training.learning_rate": training.learning_rate,
            "training.temperature": training.temperature,
            "training.num_selfplay_iterations": training.num_selfplay_iterations,
            "mcts.dirichlet_epsilon": mcts.dirichlet_epsilon,
            "mcts.num_searches": mcts.num_searches,
            "evaluation.promotion_win_rate": evaluation.promotion_win_rate,
            "evaluation.random_play_ratio": evaluation.random_play_ratio,
        }
        if training.random_play_ratio is not None:
            schedules_to_validate["training.random_play_ratio"] = (
                training.random_play_ratio
            )
        if training.opponent_rates is not None:
            schedules_to_validate["training.opponent_rates.random_bot_ratio"] = (
                training.opponent_rates.random_bot_ratio
            )
            schedules_to_validate["training.opponent_rates.prev_bot_ratio"] = (
                training.opponent_rates.prev_bot_ratio
            )
        if evaluation.eval_num_searches is not None:
            schedules_to_validate["evaluation.eval_num_searches"] = (
                evaluation.eval_num_searches
            )
        if evaluation.baseline_num_searches is not None:
            schedules_to_validate["evaluation.baseline_num_searches"] = (
                evaluation.baseline_num_searches
            )

        validate_against_num_iterations(
            schedules_to_validate,
            num_iterations=num_iterations,
        )

        paths = self.paths
        for field_name, expected in {
            "replay_dir": PATH_REPLAY_DIR,
            "ckpt_dir": PATH_CKPT_DIR,
            "evaluation_logs_dir": PATH_EVAL_LOGS_DIR,
            "manifest": PATH_MANIFEST,
        }.items():
            current = getattr(paths, field_name)
            if current and current != expected:
                raise ValueError(
                    f"paths.{field_name} is fixed to '{expected}'; remove overrides."
                )

        paths_updates: dict[str, Any] = {
            "replay_dir": PATH_REPLAY_DIR,
            "ckpt_dir": PATH_CKPT_DIR,
            "evaluation_logs_dir": PATH_EVAL_LOGS_DIR,
            "manifest": PATH_MANIFEST,
        }

        if paths.use_gcs:
            if paths.run_prefix is None:
                raise ValueError(
                    "paths.run_prefix is required when use_gcs=True (set to bucket name)."
                )
            if "/" in paths.run_prefix or paths.run_prefix.startswith("gs://"):
                raise ValueError(
                    "paths.run_prefix must be plain bucket name when use_gcs=True."
                )
        else:
            if not paths.run_prefix:
                paths_updates["run_prefix"] = PATH_RUN_PREFIX

        if not paths.run_id:
            paths_updates["run_id"] = time.strftime("%Y%m%d-%H%M%S")
        paths = paths.model_copy(update=paths_updates)

        return self.model_copy(
            update={
                "model": model,
                "training": training,
                "mcts": mcts,
                "evaluation": evaluation,
                "paths": paths,
            }
        )


class RunnerParams(BaseConfig):
    board: BoardConfig
    model: ModelConfig
    mcts: MctsConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    parallel: ParallelConfig
    paths: PathsConfig
    io: IoConfig
    runtime: RuntimeConfig | None = None
    device_type: str
    run_id: str
    config_name: str | None = None


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_config(raw: dict[str, Any]) -> RootConfig:
    return RootConfig.model_validate(raw)


def load_and_parse_config(path: str) -> RootConfig:
    raw = load_yaml(path)
    return parse_config(raw)


def assemble_runner_params(
    cfg: RootConfig, device_type: str, run_id: str, *, config_name: str | None = None
) -> RunnerParams:
    return RunnerParams(
        board=cfg.board,
        model=cfg.model,
        mcts=cfg.mcts,
        training=cfg.training,
        evaluation=cfg.evaluation,
        parallel=cfg.parallel,
        paths=cfg.paths,
        io=cfg.io,
        runtime=cfg.runtime,
        device_type=device_type,
        run_id=run_id,
        config_name=config_name,
    )
