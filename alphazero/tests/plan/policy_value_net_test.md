
# PolicyValueNet / ML 파이프라인 테스트 체크리스트 (pytest)

---

## Step 1: Feature Logic (데이터 로직 검증)
**파일:** `policy_value_net_feature_logic_test.py`

- [v] `test_model_config_uses_safe_defaults`
  - Config 누락 시 안전한 기본값이 적용되고 필수 필드가 초기화되어 실행이 중단되지 않는지 확인한다.
- [v] `test_last_move_plane_marks_correct_position`
  - Last Move 평면이 마지막 착수만 1로 표기되고 나머지는 0으로 유지되며 중복 표시가 없는지 확인한다.
- [v] `test_color_plane_turn_encoding`
  - Color 평면이 현재 턴의 흑/백을 정확히 인코딩하고 턴 전환 시 값이 즉시 바뀌어 혼동이 없는지 확인한다.
- [v] `test_capture_score_plane_normalization`
  - Capture Score 평면이 캡처 점수를 합의된 스케일로 정규화해 범위가 기대값에 머무는지 확인한다.
- [v] `test_forbidden_plane_marks_double_three`
  - Forbidden 평면이 3-3 금수 위치만 1로 표시하고 합법 수는 0으로 남아 오탐이 없는지 확인한다.
- [v] `test_history_planes_stack_and_shift_correctly`
  - History 평면이 과거 수를 시간 순서로 스택하고 새 착수마다 시프트가 정확해 누락이 없는지 확인한다.
- [v] `test_feature_consistency_after_terminal_state`
  - 게임 종료 직후에도 Last Move/Color/History 등 주요 피처가 마지막 상태와 일치하는지 확인한다.

---

## Step 2: Encoding Invariants (데이터 텐서 무결성)
**파일:** `policy_value_net_encoding_invariants_test.py`

- [v] `test_encoding_invariants_and_constraints`
  - Me/Opp/Empty 합=1, Padding=0 등 공통 불변식이 인코딩 전 구간에서 유지되는지 확인한다.

---

## Step 3: Model Architecture (모델 구조)
**파일:** `policy_value_net_architecture_test.py`

- [v] `test_forward_output_shapes`
  - Policy/Value 출력 텐서 shape가 기대한 차원과 일치해 배치 처리가 가능한지 끝까지 확인한다.
- [v] `test_value_head_range`
  - Value Head 출력이 [-1.0, 1.0] 범위 안에 머물러 과도한 saturation이 없는지 확인한다.
- [v] `test_backward_propagates_gradients`
  - Loss backward 후 모든 파라미터에 gradient가 생성되고 0이 아닌 값이 존재하는지 확인한다.
- [v] `test_parameter_count_matches_config`
  - Config 기준 파라미터 수가 기대 값과 일치해 설계된 모델 구조의 정합성이 끝까지 유지되는지 확인한다.

---

## Step 4: System Integration (시스템 통합 & 견고성)
**파일:** `policy_value_net_system_integration_test.py`

- [v] `test_variable_board_size_end_to_end`
  - 9x9~19x19 가변 보드에서 인코딩→모델 추론이 엔드투엔드로 에러 없이 이어져 호환되는지 확인한다.
- [v] `test_batch_processing_consistency`
  - 단일 추론과 배치 추론 결과가 일관되고 배치 인코딩 간 간섭이나 누수가 없어 정합성이 유지되는지 확인한다.
- [v] `test_serialization_and_device_mobility`
  - state_dict 저장/로드 후에도 CPU↔CPU 라운드트립에서 출력이 동일해 일관성이 유지되는지 확인한다.
- [ ] `test_device_mobility_and_consistency`
  - (GPU 환경에서만) 모델/입력을 CUDA로 옮겨도 출력이 동일하고 오차가 과도하지 않은지 확인한다.
- [v] `test_mismatched_input_channels_raises_error`
  - 채널 수가 맞지 않는 입력에 대해 명확한 예외가 즉시 발생하고 메시지가 디버깅에 충분한지 확인한다.
- [v] `test_seed_reproducibility_for_model_init_and_forward`
  - 시드 고정 시 모델 초기화와 Forward 결과가 재현되어 실험 간 결정론이 유지되는지 확인한다.

---

## Step 5: Learning Capability (학습 능력)
**파일:** `policy_value_net_learning_capability_test.py`

- [ ] `test_single_batch_overfit`
  - 단일 배치로 반복 학습 시 loss가 충분히 감소하고 정답에 근접해 과적합이 가능한지 확인해 학습성을 검증한다.
