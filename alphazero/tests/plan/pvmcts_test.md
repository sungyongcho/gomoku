# PVMCTS Engine / Search 테스트 체크리스트 (pytest)

---

## Step 1: Engine Common (공통 로직 – 모든 모드 필수)
**파일:** `tests/pvmcts/engine/engine_common_test.py`

- [v] `test_safe_dirichlet_noise_mechanics`
  - Legal Moves ≥ 2일 때 확률 합이 1.0을 유지하고, 불법 수 확률이 0.0으로 유지되는지 확인
- [v] `test_safe_dirichlet_noise_edge_cases`
  - epsilon=0 또는 Legal Moves ≤ 1인 경우 원본 Policy 보존 확인
- [v] `test_masked_policy_consistency`
  - Logit 크기와 무관하게 불법 수 마스킹 및 Softmax 합 1.0 유지 확인
- [v] `test_masked_policy_nan_prevention`
  - 모든 수가 불법이거나 극단값일 때 NaN 방지 처리 확인
- [v] `test_legal_mask_tensor_mapping`
  - Board 상태가 Tensor Mask로 정확히 변환되는지 확인
- [v] `test_evaluate_terminal_logic`
  - 루트 노드: check_win(패배), empty=0(무승부) 즉시 감지
  - 자식 노드: 승리(1.0) 감지 시 부모 관점 패배(-1.0) 반전 확인
- [v] `test_encode_state_shape_and_device`
  - 입력 상태가 (C, H, W) 텐서 및 올바른 장치로 변환되는지 확인
- [v] `test_backup_value_propagation_and_sign_flip`
  - 깊이마다 value 부호 반전되며 value_sum에 누적되는지 확인

---


## Step 2: Sequential Search Logic
**파일:** `tests/pvmcts/search/sequential_test.py`

### Root & Expansion
- [v] `test_start_node_expansion_and_dirichlet`
  - 초기 visit_count=0에서 확장 및 시작 노드(헤드)에만 Dirichlet 적용 확인

### Simulation Flow
- [v] `test_simulation_loop_and_selection`
  - num_searches만큼 반복, visit_count 증가 및 UCB 기반 트리 확장 확인

### Terminal Handling
- [v] `test_terminal_node_handling`
  - 터미널 도달 시 추가 추론 없이 즉시 백업되는지 확인

### Edge Cases & Robustness
- [v] `test_draw_handling_on_full_board`
  - 확장 불가 리프 재방문 대신 실제 Full-Board 무승부 처리 검증
- [v] `test_single_legal_move_processing`
  - 합법 수 1개 상황에서도 정상 탐색/백업 확인

### Inference & Determinism
- [v] `test_batch_size_ignorance`
  - SequentialEngine이 batch_size 설정을 무시하고 단일 추론만 수행하는지 확인
- [v] `test_reproducibility_with_fixed_seed`
  - 시드 고정 시 search 결과 visit 분포 재현성 확인
- [v] `test_search_result_policy_distribution`
  - Mock 추론 Policy 경향이 visit_count 분포에 반영되는지 확인



## Step 3: Vectorized Search Logic
**파일:** `tests/pvmcts/search/vectorize_test.py`

### Input & Interface
- [v] `test_vectorize_single_start_node_input`
  - 시작 노드(head/start node)에 단일 TreeNode를 전달해도 내부에서 리스트로 변환되어 정상 처리되는지 확인
- [v] `test_vectorize_empty_input`
  - 빈 리스트를 전달했을 때 크래시 없이 즉시 반환되는지 확인

### Initialization & Policy
- [v] `test_vectorize_root_dirichlet_only`
  - 시작 노드(head/start node) 확장에만 Dirichlet 노이즈가 적용되고 이후 확장에서는 순수 Softmax가 유지되는지 검증

### Simulation Flow
- [v] `test_vectorize_respects_num_searches`
  - 시작 노드(head/start node) 초기 visit_count가 혼재된 경우에도 각 노드가 num_searches를 초과하지 않는지 확인
- [v] `test_vectorize_active_set_drops_finished`
  - 모든 시작 노드(head/start node)가 done 상태가 되면 메인 루프가 즉시 종료되어 불필요한 반복이 없는지 확인

### Terminal Handling
- [v] `test_vectorize_terminal_short_circuit`
  - 터미널 시작 노드가 주어질 때 추가 추론 없이 즉시 백업·종료되는지 확인
- [v] `test_vectorize_mixed_terminal_and_active`
  - 터미널/비터미널 시작 노드를 섞어 넣었을 때 터미널 노드는 건너뛰고 나머지만 시뮬레이션되는지 검증

### Batch & Mapping
- [v] `test_vectorize_batch_inference_shapes`
  - 다수 시작 노드(start node) 입력 시 infer_batch가 한 번만 호출되고 정책/가치 인덱스가 올바르게 매핑되는지 모킹으로 검증
- [v] `test_vectorize_large_batch_processing`
  - 대규모 배치 입력을 단일 infer_batch 호출로 처리하며 모든 시작 노드 방문/확장이 정상 수행되는지 확인
- [v] `test_vectorize_fallback_to_standard_infer`
  - infer_batch가 없는 클라이언트일 때 infer가 호출되고 입력 텐서 차원이 올바르게 유지되는지 확인
- [v] `test_vectorize_enforces_cpu_tensors`
  - VectorizeEngine이 inference 호출 시 입력 텐서를 CPU 디바이스로 고정하는지 확인

### Robustness / Edge Cases
- [v] `test_vectorize_no_zero_backup_bias`
  - visit_count>0 리프 재방문 시 0.0 백업을 하지 않아 통계가 변하지 않는지 확인
- [v] `test_vectorize_handles_no_legal_moves`
  - 합법 수가 없는 노드에서 정책이 안전하게 0 처리되고 크래시가 없는지 확인
- [v] `test_vectorize_duplicate_states_in_batch`
  - 서로 다른 루트가 동일 상태를 공유할 때 배치 추론 입력/출력이 의도대로 매핑되는지 확인


## Step 4: Multiprocessing Search Logic
**파일:** `tests/pvmcts/search/mp_test.py`

### Initialization & Environment
- [v] `test_mp_engine_reseeds_random_generators`
  - 엔진 초기화 시 _seed_random_generators가 호출되어 numpy/torch 난수 시드가 부모 프로세스와 달라지는지 확인
- [v] `test_mp_engine_pid_assignment`
  - 엔진 인스턴스 생성 시 os.getpid()가 올바르게 기록되는지 확인(로깅·디버깅용)

### Batch & Mapping
- [v] `test_mp_engine_batch_mapping`
  - 다수 시작 노드 입력 시 mp 클라이언트 단일 호출로 반환된 value가 각 노드에 올바르게 매핑되는지 확인
- [v] `test_mp_engine_data_integrity_over_ipc`
  - np.stack으로 만든 배치가 (B, C, H, W)와 dtype을 유지한 채 클라이언트로 전달되는지 검증
- [v] `test_mp_engine_cpu_payload`
  - 워커로 전달되는 입력 텐서가 CPU 디바이스로 강제되는지 확인
- [v] `test_mp_engine_result_device_handling`
  - 클라이언트가 GPU 텐서를 반환해도 엔진이 .cpu()로 안전하게 내려서 처리하는지 확인

### Policy & Flow
- [v] `test_mp_engine_root_dirichlet_only`
  - 루트 확장에만 Dirichlet 노이즈가 적용되고 이후 확장은 순수 Softmax가 유지되는지 검증
- [v] `test_mp_engine_empty_input`
  - 시작 노드가 빈 리스트일 때 추론 요청 없이 조기 반환되어 불필요한 통신이 없는지 확인

### Error Handling & Robustness
- [v] `test_mp_engine_handles_broken_pipe`
  - MPInferenceClient가 BrokenPipe/EOF를 던질 때 search가 로깅 후 안전하게 종료/예외를 전파하는지 확인
- [v] `test_mp_engine_mismatched_batch_size_error`
  - 요청 배치 크기와 반환 배치 크기가 다를 때 적절히 검출/핸들링되는지 확인

---

## Step 5-1: BatchInferenceManager Unit Tests
**파일:** `tests/pvmcts/batching_manager_test.py`

### Dispatch & Queueing
- [ ] `test_enqueue_and_dispatch_by_size`
  - batch_size 도달 시 dispatch_ready가 True이고 클라이언트 호출되는지 확인
- [ ] `test_dispatch_by_timeout`
  - min_batch_size 미만이어도 max_wait_ms 경과 후 dispatch되는지 확인
- [ ] `test_backpressure_max_inflight`
  - max_inflight 초과 시 dispatch_ready가 False로 요청을 차단하는지 확인
- [ ] `test_dispatch_force`
  - min_batch_size/timeout 미충족 상태에서도 force=True로 즉시 발송되는지 확인

### Results & Counting
- [ ] `test_drain_results_matching`
  - PendingNodeInfo 순서대로 결과 매칭되는지 확인
- [ ] `test_get_pending_node_count`
  - 큐 + inflight 노드 수 합계가 정확히 반환되는지 확인
- [ ] `test_flush_clears_everything`
  - flush() 후 큐/refs가 비워지고 결과를 반환하는지 확인

---

## Step 5-2: RayAsyncEngine Logic Tests (Mock 기반)
**파일:** `tests/pvmcts/search/ray_test.py`

### Search Flow & Pipelining
- [ ] `test_exact_search_count`
  - num_searches만큼만 backup이 호출되는지 검증
- [ ] `test_pipelining_behavior`
  - 느린 infer_async에도 Selection이 진행되어 큐가 채워지는지 확인
- [ ] `test_multi_root_handling`
  - 여러 루트 입력 시 각 루트가 목표 횟수를 채우는지 확인
- [ ] `test_inflight_limit_throttling`
  - async_inflight_limit 도달 시 enqueue를 멈추는지 확인
- [ ] `test_error_handling_during_search`
  - 배치 실패 시 on_error 정책에 따라 계속 진행/예외 발생을 확인
- [ ] `test_ray_engine_mismatched_result_shape_error`
  - 반환 텐서의 배치 차원이 요청 개수와 불일치할 때 RuntimeError로 감지하는지 확인
- [ ] `test_ray_engine_terminal_node_bypassing`
  - 터미널 노드는 enqueue 없이 즉시 backup 처리되는지 검증

---

## Step 5-3: Ray Integration Tests
**파일:** `tests/pvmcts/search/ray_test.py`

- [ ] `test_ray_local_end_to_end`
  - ray.init(local_mode=True) 환경에서 end-to-end 검색이 에러 없이 완료되는지 확인
- [ ] `test_determinism_with_seed`
  - 고정 시드로 비동기 환경에서도 주요 통계가 허용 오차 내에서 재현되는지 확인


---
