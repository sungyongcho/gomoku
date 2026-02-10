## 27. 대국 결과 저장 시 관점 변환 (Relative Outcome Storage)

- **개요**: 게임 종료 후 데이터를 저장할 때, 최종 승패(Value) 하나를 일괄 적용하는 것이 아니라, 각 턴을 둔 플레이어(흑/백)의 관점에 맞춰 보상의 부호(+, -)를 반전시켜 매핑하는 로직.
- **효과**: 데이터 무결성(Data Correctness) 보장. 패배한 턴을 승리로, 혹은 그 반대로 잘못 학습하는 치명적인 오류를 방지.
- **gmk(Old) 위치**: alphazero/alphazero_parallel.py (hist_player == player 비교 후 부호 반전).
- **gmk-refactor(New) 구현 권장위치**: gomoku/alphazero/runners/common.py (build_game_record 함수)<br><br>gomoku/alphazero/runners/selfplay.py<br><br>단일 value를 받아 히스토리를 순회하며 플레이어 턴에 맞춰 value 또는 -value를 할당하는 로직 검증 및 보완.
- **상세 유형**: [Critical] Data Correctness


## 5. MCTS Root Parallelism 및 Virtual Loss (Parallel Search Safety)

- **개요**: 여러 스레드/워커가 동시에 MCTS 트리를 탐색할 때, 동일한 노드를 중복 선택하는 "충돌(Collision)"을 방지하기 위한 가상 패배(Virtual Loss) 메커니즘을 C++로 구현해 atomics 기반으로 처리.
- **효과**: 탐색 효율성 보장. 병렬 탐색 시 중복 탐색으로 인한 효율 급감 및 탐색 품질 저하를 예방하며, C++ 구현 시 락/GIL 오버헤드를 줄여 탐색 처리량을 높임.
- **gmk(Old) 위치**: alphazero/pvmcts_parallel.py (병렬 MCTS 구현체).
- **gmk-refactor(New) 구현 권장위치**: gomoku/pvmcts/treenode.py (visit_count 임시 증가 로직)<br><br>gomoku/pvmcts/search/ 하위 엔진 (Selection 단계에서 Virtual Loss 적용).
- **상세 유형**: [Critical] Search Efficiency / Correctness
- **C++ 구현 방법**: treenode 및 search 엔진을 pybind11로 래핑한 C++ 확장으로 작성하고, virtual loss/visit_count를 `std::atomic<int>`로 관리해 Selection에서 lock-free로 증가·복원하도록 한다.

## 1. 데이터 증강 (Symmetry Data Augmentation)

- **개요**: 보드 게임의 회전(90/180/270도) 및 대칭(Flip) 불변성을 이용하여, 1개의 대국 데이터를 8개로 증강하여 학습에 활용하는 로직.
- **효과**: 데이터 효율성 8배 향상. 적은 대국 수로도 모델이 기보의 패턴을 빠르게 일반화하여 학습 속도와 성능이 비약적으로 상승함.
- **gmk(Old) 위치**: alphazero/selfplay.py 등에서 데이터 저장 전 get_symmetries 호출.
- **gmk-refactor(New) 구현 권장위치**: gomoku/alphazero/learning/dataset.py<br><br>__getitem__ 메서드 내에 무작위 회전/대칭 로직을 추가하거나, 데이터 생성 시점에 증강하여 저장.
- **상세 유형**: [Critical] Data Efficiency

## 2. 학습 파라미터 스케줄링 및 전파 (Parameter Scheduling)

- **개요**: 학습 진행도(Iteration)에 따라 temperature, LR, epsilon 등을 동적으로 조절하고 이를 워커에 전파하는 기능.
- **효과**: 학습 후반부 성능 최적화. 초기에는 탐색(Exploration)을 유도하고 후반에는 미세 조정(Exploitation)을 강화하며, 정체 구간을 돌파함.
- **gmk(Old) 위치**: alphazero/alphazero_parallel.py (매 루프마다 Config 스케줄 확인 후 워커 전파).
- **gmk-refactor(New) 구현 권장위치**: gomoku/alphazero/runners/workers/ray_worker.py<br><br>run_games 호출 시 현재 iteration 정보를 인자로 전달받아 내부 mcts_cfg를 갱신하도록 수정.
- **상세 유형**: [Critical] Training Dynamics

## 20. 파라미터 변경에 따른 워커 재설정 (Worker Parameter Sync)

- **개요**: 학습 Iteration이 바뀔 때마다 변경된 하이퍼파라미터(Temperature, Exploration Noise 등)를 실행 중인 Ray Actor(워커)들에게 전송하여 갱신하는 로직.
- **효과**: 학습 스케줄링 적용. 학습 초반에는 탐색을 많이 하고 후반에는 줄이는 등의 전략(Scheduling)이 실제 워커에 반영되도록 함. (현재 New 레포는 초기값 고정 문제 있음)
- **gmk(Old) 위치**: alphazero/alphazero_parallel.py (루프 마다 set_weights와 함께 설정 전파).
- **gmk-refactor(New) 구현 권장위치**: gomoku/alphazero/runners/workers/ray_worker.py (run_games 메서드)<br><br>gomoku/alphazero/runners/ray_runner.py<br><br>run_games 호출 시 mcts_config 딕셔너리를 인자로 넘기고, 워커 내부에서 self.agent.mcts_cfg를 업데이트하는 코드 추가.
- **상세 유형**: [Critical] Training Dynamics

## 29. PER 우선순위 갱신 로직 (PER Priority Update)

- **개요**: Prioritized Experience Replay(PER) 사용 시, 학습 미니배치에서 계산된 손실(TD-Error) 값을 바탕으로 해당 샘플들의 우선순위(Priority)를 갱신하고 샘플링 가중치를 업데이트하는 로직을 C++ 확장으로 구현.
- **효과**: 알고리즘 동등성 및 성능 확보. C++ 벡터화/병렬 갱신으로 PER가 동적으로 '어려운 예제'에 가중치를 두며, 파이썬 루프 없이 대규모 버퍼에서의 우선순위 업데이트 지연을 최소화.
- **gmk(Old) 위치**: alphazero/alphazero.py (학습 스텝 후 memory.update_priorities 호출).
- **gmk-refactor(New) 구현 권장위치**: gomoku/alphazero/learning/trainer.py<br><br>train_step 메서드 완료 후 반환된 loss(또는 TD-error)를 사용하여 dataset 혹은 replay_buffer의 우선순위를 갱신하는 코드 추가.
- **상세 유형**: [Critical] Algorithm Correctness
- **C++ 구현 방법**: PER 버퍼를 관리하는 클래스를 C++로 구현하고, 우선순위 배열을 SIMD/병렬로 갱신하는 함수를 pybind11로 노출해 trainer에서 일괄 호출한다.

## 26. 탐색 온도 적용 시 수치적 안전장치 (Numerical Stability)

- **개요**: Temperature 파라미터가 0에 가까울 때(Annealing, 평가 등) 발생할 수 있는 '0으로 나누기(Division by Zero)'나 'Log(0)' 에러를 방지하기 위한 최소값 클리핑(Clipping) 로직.
- **효과**: 시스템 안정성 확보. 학습 중 Temperature가 변할 때 런타임 크래시를 방지하고 안정적인 확률 분포 계산 보장.
- **gmk(Old) 위치**: alphazero/selfplay.py (max(1e-8, temp) 및 np.maximum 사용).
- **gmk-refactor(New) 구현 권장위치**: gomoku/alphazero/agent.py (get_action_probs 메서드)<br><br>gomoku/alphazero/runners/selfplay.py<br><br>입력받은 temperature 값에 대해 max(epsilon, temp) 처리를 수행하는 방어 코드 추가.
- **상세 유형**: [Low] Stability

## 25. MCTS 미방문 노드 초기값 처리 (FPU Logic)

- **개요**: MCTS 탐색 시 방문 횟수(visit_count)가 0인 리프 노드의 가치(Q-value)를 초기화하는 명확한 로직(예: 0.0, 무한대, 부모 값 상속 등)을 C++로 구현해 Selection 단계의 분기 비용을 최소화.
- **효과**: 탐색 성능 보장. 미방문 노드에 대해 적절한 우선순위를 부여함으로써 탐색의 균형(Exploration vs Exploitation)을 맞추고 수렴 속도 저하를 방지하며, 네이티브 코드로 분기·계산 비용을 절감. (현재는 기본값 0.0으로 동작)
- **gmk(Old) 위치**: alphazero/pvmcts.py (get_ucb 메서드 내 0.0으로 명시적 초기화).
- **gmk-refactor(New) 구현 권장위치**: gomoku/pvmcts/treenode.py<br><br>gomoku/pvmcts/search/engine.py<br><br>Node 클래스나 엔진의 Select 단계에서 visit_count == 0인 경우의 반환 값을 명시적으로 처리.
- **상세 유형**: [Medium] Search Algorithm
- **C++ 구현 방법**: UCB 계산과 FPU 초기화를 C++ 인라인 함수로 구현하고, 노드 구조체에 초기 Q 값을 상수로 보관하여 visit_count가 0일 때 분기 없이 반환하도록 작성한다.

## 14. 완전 무작위 수 (Explicit Random Play)

- **개요**: MCTS 정책과 무관하게, 설정된 확률(epsilon)로 완전히 무작위 수를 두는 Epsilon-Greedy 탐색.
- **효과**: 탐색 다양성 확보. 국소 최적해(Local Optima) 탈출 및 다양한 오프닝 데이터 확보.
- **gmk(Old) 위치**: alphazero/selfplay.py 내 random_play_ratio 확인 로직.
- **gmk-refactor(New) 구현 권장위치**: gomoku/alphazero/runners/selfplay.py 또는 agent.py<br><br>Action 선택 시 random.random() < ratio 분기 추가.
- **상세 유형**: [Low] Exploration

## 21. MCTS Candidate Masking 옵션 (Search Pruning)

- **개요**: 현재 착수된 돌들로부터 일정 거리(예: 2칸) 이상 떨어진 곳은 탐색 후보에서 아예 제외하는 최적화 기법 및 이를 켜고 끄는 Config 옵션을 C++로 구현해 비트마스크 연산으로 가볍게 처리.
- **효과**: 탐색 속도 및 효율 극대화. 15x15 보드에서 불필요한 영역(돌이 없는 구석 등)을 탐색하는 계산 낭비를 줄여, 동일 시간 내 더 깊은 수읽기 가능하며, C++ 비트 연산으로 추가 오버헤드를 최소화.
- **gmk(Old) 위치**: core/gomoku.py (_build_candidate_mask 및 관련 Config).
- **gmk-refactor(New) 구현 권장위치**: gomoku/core/gomoku.py (get_legal_moves 메서드)<br><br>gomoku/core/game_config.py<br><br>use_candidate_mask 옵션을 Config에 추가하고, get_legal_moves에서 거리 기반 필터링 로직 복원.
- **상세 유형**: [High] Search Efficiency
- **C++ 구현 방법**: 보드 상태를 비트보드로 유지하고, 최근 착수 주변을 dilation해 후보를 생성하는 함수를 C++로 작성 후 pybind11로 노출해 get_legal_moves에서 호출한다.

## 8. 후보 수 마스킹 (Candidate Masking)

- **개요**: 돌이 놓일 가능성이 없는(기존 돌에서 멀리 떨어진) 위치를 탐색 후보에서 제외하는 휴리스틱을 C++ 네이티브 함수로 구현해 반복 호출 비용을 낮춤.
- **효과**: 탐색 속도 증대. MCTS가 불필요한 영역을 탐색하는 낭비를 줄이며, 파이썬 루프를 제거해 대량 호출에서도 성능을 확보.
- **gmk(Old) 위치**: core/gomoku.py의 _build_candidate_mask.
- **gmk-refactor(New) 구현 권장위치**: gomoku/core/gomoku.py<br><br>get_legal_moves 내부에 거리 기반 마스킹 로직 복구 (Config로 On/Off 옵션화).
- **상세 유형**: [Medium] Search Pruning
- **C++ 구현 방법**: 보드 배열을 전달받아 맨해튼 거리 기반 필터를 적용하는 함수를 C++로 작성하고, PyTorch 텐서/NumPy 배열을 직접 받아 불리언 마스크를 반환하도록 바인딩한다.

## 10. Evaluation 시 Temperature 고정 문제 (Evaluation Reliability)

- **개요**: 평가 대국 시 초반 수순(Opening) 이후에는 결정론적(Deterministic, Temp=0)으로 두어야 모델의 정확한 실력을 평가 가능.
- **효과**: 평가 신뢰성 확보. 평가 결과의 무작위성을 제거하여 승률 지표의 변동성을 줄임.
- **gmk(Old) 위치**: alphazero/arena_runner.py (Opening 이후 Argmax 사용).
- **gmk-refactor(New) 구현 권장위치**: gomoku/alphazero/eval/arena.py<br><br>match.py 호출 시 temperature 인자 전달 로직 점검. Opening 이후에는 강제로 Temp=0(Argmax)이 되도록 수정.
- **상세 유형**: [Medium] Evaluation Correctness

## 16. Baseline 평가 및 승격 게이트 (Promotion Gate Logic)

- **개요**: 새로운 모델이 승격되기 위해 챔피언 모델뿐만 아니라, 기준점(Baseline) 모델도 일정 승률 이상 이겨야 하며, 동시에 '블런더(치명적 실수) 비율'이 허용치 이내여야 한다는 복합 승격 조건.
- **효과**: 모델 품질 안전장치. 챔피언과의 승률만으로 판단할 때 발생하는 가위바위보 상성 문제(Cycle)나, 승률은 좋지만 어이없는 실수(Blunder)가 늘어나는 품질 저하를 방지함.
- **gmk(Old) 위치**: alphazero/arena_runner.py (승격 결정 로직 내 Baseline/Blunder 복합 체크).
- **gmk-refactor(New) 구현 권장위치**: gomoku/alphazero/eval/arena.py (evaluate_model 메서드)<br><br>단순 승률 비교(win_rate > threshold) 로직을 check_promotion_criteria(metrics, baseline_metrics) 형태의 함수로 확장하여 복합 조건 검사 추가.
- **상세 유형**: [High] Evaluation Reliability

## 18. Soft SPRT 및 Fallback 결정 (Adaptive Evaluation)

- **개요**: SPRT(순차적 확률비 검사)를 적용하되, 엄격한 통과/탈락 외에 'Soft Margin'(약간의 승률 우위)을 인정하거나, 최대 게임 수 도달 시 승률에 따라 조건부 승격(Fallback Accept)하는 유연한 평가 로직.
- **효과**: 평가 효율성 및 공정성. 압도적이지 않은 개선 사항도 놓치지 않고 반영하며, 무의미한 무승부성 대국을 조기에 종료하여 평가 리소스를 절약.
- **gmk(Old) 위치**: alphazero/arena_runner.py (SPRT 종료 후 Fallback 로직).
- **gmk-refactor(New) 구현 권장위치**: gomoku/alphazero/eval/arena.py (_run_duel 및 evaluate_model)<br><br>gomoku/alphazero/eval/sprt.py<br><br>SPRT 미결정(Indeterminate) 상태로 루프 종료 시, fallback_accept_threshold를 확인하는 분기 추가.
- **상세 유형**: [High] Evaluation Efficiency

## 19. Baseline 평가 결과의 Score Rate 및 ELO 변환

- **개요**: Baseline 모델과의 대국 결과를 단순 승률(Win Rate)이 아닌, 무승부를 포함한 Score Rate로 정규화하고 이를 기반으로 상대적 ELO 점수를 산출하여 지표로 활용.
- **효과**: 정밀한 성능 지표. 단순 승/패 승률보다 무승부가 섞인 상황에서 모델의 실력을 더 정확하게 수치화하여 비교 가능.
- **gmk(Old) 위치**: alphazero/arena_runner.py (결과 집계 및 ELO 계산).
- **gmk-refactor(New) 구현 권장위치**: gomoku/alphazero/eval/metrics.py (H2HMetrics 클래스 확장)<br><br>gomoku/alphazero/eval/arena.py<br><br>승률 계산 시 (wins + 0.5 * draws) / total 공식을 적용하고, 이를 ELO 공식에 대입하는 유틸리티 추가.
- **상세 유형**: [Medium] Metrics Accuracy

## 9. 평가 기보 저장 (Evaluation Logging)

- **개요**: Arena 평가 대국의 상세 내용(수순, 정책, 가치, 승부처 등)을 파일(SGF/JSON)로 저장.
- **효과**: 디버깅 및 원인 분석. 모델이 왜 졌는지(Blunder), 어떤 패턴에 취약한지 사후 분석 가능.
- **gmk(Old) 위치**: alphazero/arena_runner.py (대국 종료 후 로그 저장).
- **gmk-refactor(New) 구현 권장위치**: gomoku/alphazero/eval/arena.py, match.py<br><br>대국 종료 후 history를 수집하여 evaluation_logs_dir에 파일로 저장하는 로직 추가.
- **상세 유형**: [Medium] Debugging & Analysis

## 17. 평가 및 셀프플레이 이벤트 로깅 (Event Logging)

- **개요**: 게임 시작/종료, 예외 발생, 워커 상태 등 상세한 이벤트를 텍스트 로그 파일이나 클라우드 스토리지(GCS)에 기록하여 추적하는 기능.
- **효과**: 장애 분석 및 모니터링. 단순히 학습 지표(Loss, Winrate)만 남기는 것이 아니라, 시스템이 멈추거나 이상 행동을 할 때 원인을 파악할 수 있는 근거 데이터 확보.
- **gmk(Old) 위치**: utils/logger.py, run_ray.py 등에서 DebugLogger 호출.
- **gmk-refactor(New) 구현 권장위치**: gomoku/utils/ray/ray_logger.py<br><br>gomoku/alphazero/runners/workers/ray_worker.py<br><br>현재는 메트릭(숫자) 로깅에 집중되어 있으므로, log_event(level, message, context) 형태의 구조화된 텍스트 로깅 기능 보강.
- **상세 유형**: [Medium] Debugging & Ops

## 3. Manifest Config Revision 관리 (Configuration Tracking)

- **개요**: 학습 중단 후 재개하거나 설정을 변경했을 때, 변경 이력을 manifest.json에 기록하여 실험의 재현성을 보장하는 기능.
- **효과**: 실험 재현성 보장. 학습 도중 하이퍼파라미터(LR, MCTS 설정 등)를 변경했을 때 이력이 소실되는 것을 방지.
- **gmk(Old) 위치**: scripts/train.py (설정 변경 감지 시 Revision 추가).
- **gmk-refactor(New) 구현 권장위치**: gomoku/scripts/pipelines/run_loop.py (_update_manifest_progress)<br><br>상태 업데이트 로직에 Config 변경 감지 및 Revision 추가 로직 보강.
- **상세 유형**: [High] Experiment Reproducibility

## 4. 옵티마이저 상태 복구 (Optimizer State Resume)

- **개요**: 학습 중단 후 재개(resume) 시, 모델 가중치뿐만 아니라 옵티마이저(Adam)의 내부 상태(Momentum, Variance)까지 복구하는 기능.
- **효과**: 학습 안정성 보장. 재학습 시 모멘텀 소실로 인해 Loss가 튀거나 수렴이 지연되는 현상을 방지.
- **gmk(Old) 위치**: scripts/train.py (체크포인트에서 optim 키 로드).
- **gmk-refactor(New) 구현 권장위치**: gomoku/scripts/train.py, gomoku/utils/paths.py<br><br>체크포인트 딕셔너리에서 optimizer_state_dict를 로드하여 적용하는 로직 추가.
- **상세 유형**: [High] Training Stability

## 6. 핫 리로딩 (Hot Reloading)

- **개요**: 추론 프로세스(Inference Server)를 재시작하지 않고, 실행 중에 제어 신호(__RELOAD__)를 받아 최신 모델 가중치(state_dict)만 교체하는 기능.
- **효과**: 운영 효율성 극대화. 프로세스 재시작 오버헤드를 제거하여 중단 없는 자가 대국(Continuous Self-play) 가능.
- **gmk(Old) 위치**: mp/inference_server.py (run 루프 내 __RELOAD__ 시그널 처리).
- **gmk-refactor(New) 구현 권장위치**: gomoku/inference/mp_server.py<br><br>BatchInferenceServer 루프에 제어 큐(command_q) 확인 로직 추가 및 모델 가중치 로드 함수 연결.
- **상세 유형**: [Critical] Operation Efficiency

## 7. Ray Actor 리소스 옵션 하드코딩 수정 (Infrastructure)

- **개요**: Ray Actor 생성 시 Config의 actor_num_cpus, num_gpus 설정을 반영하여 CPU/GPU 할당을 유연하게 제어.
- **효과**: 리소스 할당 최적화. 다양한 클러스터 환경(GPU 유무 등)에 맞춰 인프라 리소스를 올바르게 활용.
- **gmk(Old) 위치**: ray.remote(**args) 또는 .options() 메서드 활용.
- **gmk-refactor(New) 구현 권장위치**: gomoku/inference/ray_client.py<br><br>RayInferenceActor 데코레이터의 제약 제거 및 Client에서 Actor 생성 시 .options(num_cpus=..., num_gpus=...) 동적 주입.
- **상세 유형**: [Medium] Resource Allocation

## 28. 학습 루프 상태 복구 및 승격 파이프라인 (Training State Recovery & Promotion Pipeline)

- **개요**: 학습 중단 후 재시작 시 기존 리플레이 버퍼(Shards)를 로드하고, 로컬 캐시를 관리하며, 챔피언 모델의 경로를 복구하여 승격/평가 루프를 끊김 없이 이어가는 운영 유틸리티 기능.
- **효과**: 운영 연속성 보장. 서버 재시작이나 크래시 발생 시에도 데이터 손실 없이 학습 상태를 온전히 복구하고, 올바른 챔피언 모델을 기준으로 평가 루프가 동작하도록 함.
- **gmk(Old) 위치**: scripts/mode/common.py (기존 샤드 로딩, 챔피언 경로 탐색 및 승격 상태 관리 로직).
- **gmk-refactor(New) 구현 권장위치**: gomoku/scripts/pipelines/common.py (신규 생성 권장)<br><br>gomoku/scripts/pipelines/run_loop.py 등 각 실행 스크립트<br><br>common.py를 신설하여 상태 복구 함수를 이식하고, run_*.py 스크립트 시작 부분에서 이를 호출하도록 구현.
- **상세 유형**: [High] Operations & Infra

## 24. 학습 루프 내 대국 상대 다양성 스케줄링 (Opponent Diversity)

- **개요**: 학습 루프 내에서 '순수 자가 대국(Self-Play)' 데이터만 수집하는 것이 아니라, 설정된 비율에 따라 '랜덤 봇과의 대국'을 수행하여 ReplayBuffer에 데이터를 혼합하는 오케스트레이션 로직.
- **효과**: 학습 강건성(Robustness) 향상. 모델이 자가 대국 패턴에만 과적합(Overfitting)되는 것을 막고, 예측 불가능한 수에 대한 대응력을 기름.
- **gmk(Old) 위치**: alphazero/alphazero_parallel.py (learn 메서드 내 selfPlay_vs_random 호출 및 비율 계산).
- **gmk-refactor(New) 구현 권장위치**: gomoku/scripts/pipelines/run_loop.py<br><br>gomoku/alphazero/learning/trainer.py<br><br>메인 루프에서 Random Opponent Task를 별도로 생성하고 생성된 기보를 버퍼에 병합하는 스케줄링 로직 추가.
- **상세 유형**: [High] Training Dynamics

## 22. 랜덤 플레이 비율 스케줄링 (Random Play Schedule)

- **개요**: Self-play 데이터 생성 시, MCTS 결과와 무관하게 완전히 무작위 수를 두는 비율(random_play_ratio)을 학습 진행도에 따라 점진적으로 줄여나가는 기능.
- **효과**: 데이터 다양성 제어. 학습 초기에는 다양한 오프닝 데이터를 확보하고, 후반에는 정교한 데이터를 모으는 균형 조절.
- **gmk(Old) 위치**: alphazero/selfplay.py (Config 스케줄에 따라 ratio 변경).
- **gmk-refactor(New) 구현 권장위치**: gomoku/alphazero/runners/selfplay.py<br><br>gomoku/alphazero/runners/workers/ray_worker.py<br><br>5번 항목(파라미터 전파)과 연계하여, run_games 시점에 현재 Iteration에 맞는 random_ratio 값을 주입받아 적용.
- **상세 유형**: [Medium] Exploration Strategy

## 15. 초고속 랜덤 워커 (Random Play Worker)

- **개요**: 신경망을 거치지 않고 Numpy만으로 동작하는 경량 워커를 C++로 구현해 보드 상태 갱신과 합법 수 샘플링을 네이티브에서 처리.
- **효과**: Cold Start 가속. 학습 초기에 리플레이 버퍼를 빠르게 채울 때 유리하며, C++ RNG와 비트보드 연산으로 초당 게임 수를 극대화.
- **gmk(Old) 위치**: mp/ray_random_worker.py.
- **gmk-refactor(New) 구현 권장위치**: gomoku/alphazero/agent.py<br><br>inference_client를 타지 않고 랜덤 인덱스를 바로 리턴하는 bypass 모드 추가.
- **상세 유형**: [Low] Cold Start
- **C++ 구현 방법**: 보드 표현과 합법 수 생성, 무작위 선택을 C++ 모듈로 작성해 pybind11로 노출하고, Ray 워커에서 해당 모듈을 호출해 플레이 시뮬레이션을 수행한다.

## 11. 승패 조기 판정 (Adjudication)

- **개요**: 승률(Value)이 임계값(예: 95%~99%)을 넘으면 게임을 즉시 종료하고 승패를 확정.
- **효과**: 리소스 절약. 이미 기운 게임을 끝까지 두는 시간 낭비 방지.
- **gmk(Old) 위치**: alphazero/arena_runner.py 또는 match 로직.
- **gmk-refactor(New) 구현 권장위치**: gomoku/alphazero/eval/match.py (play_match)<br><br>루프 내 root_value > adjudication_threshold 검사 후 조기 리턴 추가.
- **상세 유형**: [Low] Resource Saving

## 12. 2단계 약식 평가 (Fast Eval)

- **개요**: 전체 평가(예: 240판) 전, 적은 판수(예: 40판)로 1차 검증하여 성능 미달 모델을 미리 탈락(Reject).
- **효과**: 평가 시간 단축. 성능이 낮은 모델에 대한 불필요한 전체 평가를 생략하여 GPU 자원 절약.
- **gmk(Old) 위치**: alphazero/arena_runner.py 내 분기 처리.
- **gmk-refactor(New) 구현 권장위치**: gomoku/alphazero/eval/arena.py (run_arena)<br><br>fast_eval 옵션 활성화 시 약식 대국 선행 및 승률 체크 로직 추가.
- **상세 유형**: [Low] Resource Saving

## 13. 최대 연패 지표 (Max Loss Streak)

- **개요**: 평가 중 모델이 연속으로 몇 번 졌는지 기록.
- **효과**: 모델 안정성 모니터링. 승률은 비슷해도 연패가 길면 특정 패턴에 취약함을 의미.
- **gmk(Old) 위치**: alphazero/arena_runner.py의 current_loss_streak.
- **gmk-refactor(New) 구현 권장위치**: gomoku/alphazero/eval/metrics.py<br><br>H2HMetrics 클래스에 max_loss_streak 필드 및 업데이트 로직 추가.
- **상세 유형**: [Low] Monitoring
