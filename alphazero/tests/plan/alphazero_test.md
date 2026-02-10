# AlphaZero 리팩토링 테스트 플랜

---

## Phase 1: 워킹 스켈레톤 (Sequential 엔진, 통합안)
**Agent/SelfPlay/Temperature/Noise/Guard** — `tests/alphazero/agent_selfplay_test.py`
- [v] `test_selfplay_runs_one_game`
  - 자가 대국 루프가 초기화부터 종료 판정, 기록 저장까지 예외 없이 진행되고 무한 루프가 없는지 확인한다.
- [v] `test_policy_sum_and_nan_free`
  - 정책 확률 벡터가 정규화되어 합이 1에 가깝고 NaN/Inf가 없는지 여러 샘플에서 확인해 수치 안정성을 확보한다.
- [v] `test_agent_policy_masking_logic`
  - 불법 착수가 확률 0으로 마스킹되고 나머지 확률이 다시 1로 정규화되어 분포 왜곡이 없는지 확인한다.
- [v] `test_outcome_sign_by_player`
  - 종료 결과를 턴 플레이어 관점으로 부호화해 승자는 +1, 패자는 -1, 무승부는 0이 되는지 확인한다.
- [v] `test_players_moves_length_match`
  - players와 moves 길이가 같아 각 턴의 플레이어와 착수가 1:1로 대응하며 누락이 없는지 확인한다.
- [v] `test_action_and_policy_shapes`
  - 정책 텐서 shape가 (T, action_size)이고 moves가 유효 범위에 있어 인덱스 오류가 없는지 확인한다.
- [v] `test_noise_toggle_changes_pi`
  - add_noise 토글에 따라 정책 분포가 변해 결정론/비결정론 동작이 명확히 구분되는지 확인한다.
- [v] `test_temperature_near_zero_yields_one_hot`
  - 온도≈0에서 정책 분포가 one-hot에 수렴해 확률 질량이 한 수에 몰리고 다른 수는 0에 가까운지 확인한다.
- [v] `test_temperature_one_preserves_distribution`
  - 온도=1에서 입력 정책 분포가 왜곡 없이 유지되고 각 확률 비율과 합이 변하지 않아 일관성이 유지되는지 확인한다.
- [v] `test_deterministic_when_noise_off`
  - add_noise=False + 고정 시드에서 반복 실행해도 동일 pi가 재현돼 결정론이 보장되는지 확인한다.
- [v] `test_pi_differs_when_noise_on`
  - add_noise=True + 고정 시드에서도 Dirichlet로 pi가 변화해 탐색 다양성이 생기는지 확인한다.
- [v] `test_update_root_reuses_child_or_resets`
  - 업데이트한 수가 자식으로 있으면 루트를 재사용하고 없으면 리셋되어 탐색 트리가 일관되게 유지되는지 확인한다.
- [v] `test_root_resets_on_state_mismatch`
  - 외부 상태와 내부 루트가 불일치하면 즉시 리셋되어 잘못된 재사용과 탐색 상태 오염이 없는지 확인한다.

**Dataset/Trainer/Overfit/ModelDevice** — `tests/alphazero/training_test.py`
- [v] `test_flatten_length_matches_moves`
  - flatten 결과 길이가 전체 moves 수와 일치해 샘플 누락이나 중복이 없는지 확인하고 정합성을 보장한다.
- [v] `test_replay_dataset_shapes`
  - ReplayDataset가 state/policy/value 텐서 shape를 기대한 형태로 반환하는지 확인한다.
- [v] `test_decode_board_policy_bytes_and_array`
  - 보드/정책이 bytes나 ndarray로 주어져도 동일한 배열로 복원되어 직렬화 호환이 유지되는지 확인한다.
- [v] `test_train_one_iteration_updates_weights`
  - 학습 이터레이션 후 모델 가중치가 실제로 변경되어 역전파와 옵티마이저가 손실 감소 방향으로 동작하는지 확인한다.
- [v] `test_train_one_iteration_no_prefetch_error`
  - num_workers=0 환경에서 DataLoader가 prefetch 설정 없이도 에러 없이 동작하는지 확인한다.
- [v] `test_loss_metrics_are_averaged`
  - loss/policy/value 메트릭이 배치 평균으로 계산되어 샘플 수에 일관되게 반영되는지 확인한다.
- [v] `test_single_batch_overfits_loss_drops`
  - 극소 샘플을 반복 학습했을 때 loss가 충분히 감소해 단일 배치 과적합이 가능한지 확인해 학습 능력을 검증한다.
- [v] `test_optimizer_uses_config_params`
  - optimizer가 config의 learning_rate와 weight_decay를 정확히 반영하는지 확인한다.
- [v] `test_channels_last_tf32_toggles`
  - channels_last와 TF32 토글 여부에 따라 학습 경로가 정상 동작하고 예외가 없는지 확인한다.

**Main 스모크** — `tests/alphazero/main_smoke_test.py`
- [v] `test_main_runs_with_config_test`
  - config_test.yaml로 iterations=1 실행 시 파이프라인이 끝까지 예외 없이 동작하는지 확인한다.
- [v] `test_buffer_skip_then_train`
  - 버퍼가 부족할 때 학습을 건너뛰고, 이후 충분해지면 학습 단계로 진입해 흐름이 전환되는지 확인한다.
- [v] `test_temperature_schedule_passed_to_selfplay`
  - 스케줄된 온도 값이 SelfPlay에 전달되어 턴별 temperature 설정이 반영되는지 확인한다.

---

## Phase 2: Vectorize Runner (배치 자가 대국)
**VectorizeRunner/Batching Logic** — tests/alphazero/vectorize_runner_test.py
- [v] `test_run_batch_games_returns_correct_count`
  - batch_size=4 실행 시 반환된 GameRecord 리스트 길이가 4인지 확인한다.
- [v] `test_noise_flag_only_on_turn_zero`
  - turn=0 슬롯만 add_noise_flags=True이고 진행 슬롯은 False로 전달되는지 Mock으로 검증한다.
- [v] `test_finished_games_are_removed_correctly`
  - 서로 다른 턴에 끝나는 게임이 인덱스 에러 없이 제거되고 누락이 없는지 확인한다.
- [v] `test_sampling_temperature_logic`
  - 탐험 구간에는 확률 샘플링, 이후에는 argmax가 적용되는지 확인한다.
- [v] `test_update_root_batch_called_with_correct_actions`
  - 매 턴 update_root_batch가 해당 턴 액션 리스트를 정확히 받는지 확인한다.
- [v] `test_records_integrity`
  - GameRecord의 states/policies/moves/outcomes 정합성이 유지되는지 확인한다.
- [v] `test_empty_slots_short_circuit`
  - batch_size=0 등 빈 입력에서 agent 호출 없이 빠르게 반환되는지 확인한다.
- [v] `test_policy_sum_and_nan_free_batch`
  - 배치 정책마다 합≈1, NaN/Inf 없는지 확인해 수치 안정성을 검증한다.
- [v] `test_illegal_moves_masked_batch`
  - 비합법 위치 확률이 0이고 moves가 action_size 범위 내인지 확인한다.
- [v] `test_temperature_pass_through_batch`
  - temperature 인자가 get_action_probs_batch에 그대로 전달되는지 확인한다.
- [v] `test_noise_group_split_behavior`
  - noise on/off 섞일 때 그룹 분리 호출이 기대대로 이뤄지는지 Mock으로 확인한다.
- [v] `test_partial_games_continue`
  - 일부 슬롯만 종료돼도 나머지 슬롯이 다음 턴 정상 진행되는지 확인한다.
  - [v] `test_outcome_sign_by_player_batch`
    - 마지막 착수자 기준 outcomes 부호가 +1/-1/0 일관성을 유지하는지 확인한다.

## Phase 3: MP Runner
- `tests/alphazero/mp_runner_test.py`
- [v] `test_server_class_collects_and_processes_full_batch`
  - 프로세스 없이 BatchInferenceServer 메서드를 직접 호출해 배치가 꽉 찼을 때 즉시 처리되는지 검증한다.
- [v] `test_server_handles_partial_batch_on_timeout`
  - batch_size보다 적은 요청만 있을 때 타임아웃 후 부분 배치가 정상 처리되는지 확인한다.
- [v] `test_mp_runner_returns_expected_game_count`
  - num_workers × games_per_worker만큼 GameRecord가 반환되는지 확인한다.
- [v] `test_worker_uses_sequential_engine_with_mp_client`
  - 워커 에이전트가 engine_type="sequential"+MPInferenceClient를 사용하는지 검증한다.
- [v] `test_outcomes_sign_correct_per_player_mp`
  - MP 경로에서도 마지막 착수자 기준 outcomes 부호가 일관(+1/-1/0)인지 확인한다.
- [v] `test_request_response_id_alignment`
  - worker_id/request_id가 응답에서 일치하는지, 불일치 시 예외/로그가 발생하는지 확인한다.
- [v] `test_graceful_shutdown_on_none_sentinel`
  - request_q에 None을 넣으면 서버/워커가 예외 없이 종료되는지 확인한다.
- [v] `test_timeout_and_partial_batch_processing`
  - max_batch_wait_ms 설정 시 부분 배치도 정상 처리되고 레이턴시가 비정상적으로 늘지 않는지 확인한다.
- [v] `test_seed_divergence_across_workers`
  - 워커별 시드 설정으로 동일 수열이 반복되지 않는지(예: move 시퀀스 비교) 확인한다.

---

## Phase 5: Ray Runner
- `tests/alphazero/ray_runner_test.py`
- [v] `test_ray_client_roundtrip_single_batch`
  - RayInferenceActor 경로가 정책/가치 텐서를 올바른 shape로 반환해 round-trip이 유지되는지 확인한다.
- [v] `test_batch_manager_timeout_flush`
  - 큐에 일부만 쌓인 상태에서도 max_wait_ms가 지나면 강제 발송되어 결과를 정상 수집하는지 확인한다.
- [v] `test_inflight_limit_backpressure`
  - max_inflight_batches 초과 시 큐에 대기시켰다가 inflight 해소 후 다시 발송되는지 확인한다.
- [v] `test_inflight_limit_applied_in_enqueue`
  - max_inflight_batches 초과 시 enqueue가 발송을 미루고 큐에 보류해 백프레셔가 적용되는지 검증한다.
- [v] `test_ray_async_runner_smoke_generates_records`
  - RayAsyncRunner가 소량 게임을 돌려 GameRecord 리스트를 반환하고 예외 없이 종료되는지 스모크 수준으로 확인한다.
- [v] `test_pending_flush_prevents_deadlock`
  - 단일 요청 반복 상황에서도 check_and_flush가 타임아웃 후 발송해 무한 대기가 발생하지 않는지 확인한다.
- [v] `test_batch_result_mapping_integrity`
  - 서로 다른 입력을 보냈을 때 N번째 결과가 N번째 요청(TreeNode)와 정확히 매핑되는지 확인해 정책/가치가 뒤섞이지 않는지 검증한다.
- [v] `test_ray_client_distributes_requests_round_robin`
  - 여러 Actor를 띄웠을 때 클라이언트가 라운드로빈으로 균등 분배하는지, 특정 Actor에만 몰리지 않는지 검증한다.
- [v] `test_manager_cleanup_cancels_inflight_tasks`
  - 결과 수신 전 cleanup 호출 시 ray.cancel이 실행되어 pending task가 남지 않는지 확인한다.


## Phase 6: 데이터 엔지니어링
- 파일: `tests/alphazero/data_engineering_test.py`
- [v] `test_parquet_roundtrip_rows_to_shard_dataset`
  - GameRecord→rows→Parquet→ShardDataset 경로로 저장/로딩 시 state/policy/value/priority가 일관되게 복원되는지 확인한다.
- [v] `test_replay_dataset_priority_weights_exist`
  - ReplayDataset/ShardDataset가 priority 텐서를 반환하고 0/NaN 방어가 적용되는지 확인한다.
- [v] `test_per_sampler_probability_proportional_to_priority`
  - PER sampler가 (p+ε)^α 비례 확률로 샘플을 선택하는지 확률 근사로 검증한다.
- [v] `test_replay_dataset_priority_safety`
  - priority가 NaN/0이어도 안전하게 보정되어 양수 텐서로 반환되는지 확인한다.
- [v] `test_importance_correction_applied`
  - PER 활성화 시 importance 보정(1/(N·prob))^β가 손실 가중에 적용되는지 확인한다.
- [v] `test_replay_buffer_capacity_and_eviction`
  - 버퍼 최대 용량을 초과할 때 오래된 샘플이 제거되고 최신 데이터만 유지되는지 확인한다.
- [v] `test_shard_dataset_iterates_multiple_files`
  - 여러 Parquet 샤드가 있을 때 ShardDataset이 모든 샘플을 끊김 없이 순회하는지 확인한다.
- [v] `test_metadata_config_snapshot_persistence`
  - 저장된 Parquet 메타데이터에 config_snapshot이 포함되어 원본 설정을 추적할 수 있는지 확인한다.


## Phase 7: 평가/승격
- 파일: `tests/alphazero/arena_test.py`
- [v] `test_arena_promote_flag_by_winrate_and_baseline`
  - promotion_win_rate/기본 승률/블런더 한도 기준으로 promote 플래그가 올바르게 계산되는지 확인한다.
- [v] `test_arena_sprt_early_stop_accept_reject`
  - SPRT 설정에서 누적 승률에 따라 accept_h1/h0로 조기 종료되는지 모의 데이터를 통해 검증한다.
- [v] `test_arena_blunder_qdrop_uses_chosen_action`
  - 선택한 action_idx 자식의 Q값으로 q_drop을 계산해 Blunder가 은폐되지 않는지 확인한다.
- [v] `test_arena_mcts_recreated_per_game`
  - 매 게임 새 MCTS가 생성되어 트리 오염 없이 독립적으로 동작하는지 확인한다(생성 호출 수 등으로 검증).
- [v] `test_arena_swaps_colors_fairly`
  - 정해진 판수에서 흑/백 배정이 공정하게 번갈아 이뤄지는지 확인한다.
- [v] `test_sprt_calculation_includes_draws_correctly`
  - ignore_draws=False에서 무승부를 포함한 표본 수로 LLR이 계산되는지 회귀 테스트로 검증한다.
- [v] `test_eval_handles_illegal_move_as_forfeit`
  - 불법 착수 발생 시 평가 루프가 즉시 패배 처리하고 중단 없이 진행되는지 확인한다.
