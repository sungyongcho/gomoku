비동기 파이프라인 MCTS 학습 가이드 (현행 코드 정합)

이 문서는 “비동기 파이프라인 MCTS”의 개념을 단계적으로 설명하면서, 현재 코드베이스와 정확히 매칭되는 위치를 함께 짚습니다. 읽는 동안 코드와 오가는 핵심 기호/함수명을 바로 대조할 수 있도록 작성했습니다.

대상 코드
- 엔진/배치: `gomoku/pvmcts/search/ray/ray_async.py`, `gomoku/pvmcts/search/ray/batch_inference_manager.py`
- 인퍼런스: `gomoku/inference/ray_client.py`
- 러너/에이전트: `gomoku/alphazero/agent.py`, `gomoku/alphazero/runners/{selfplay,vectorize_runner,ray_runner,workers/ray_worker}.py`

---

## 1. 왜 비동기 파이프라인인가?
### 문제
- 원격 GPU/Ray actor에 배치를 던지고 결과를 “기다리는” 동기 방식(Stop-and-Wait)은 네트워크/추론 지연이 곧바로 병목이 됩니다.

### 해결
- **파이프라인**: 선택→확장→추론→백업 단계를 겹쳐서 수행합니다. GPU가 배치 추론을 하는 동안 CPU는 다음 리프를 고르고 트리를 확장합니다.
- **리프 병렬화**: 하나의 트리에서 여러 리프를 찾아 배치로 추론(GPU 벡터화 활용).
- **가상 방문(virtual loss/pending_visits)**: 비동기 요청 중인 경로를 다시 선택하지 않도록 임시 방문수를 올려 중복 선택을 방지합니다.
- **루트 노이즈**: 루트(또는 재사용 루트)에서만 Dirichlet 노이즈를 넣어 초기 다양성을 확보하고, 깊은 노드에는 노이즈를 주지 않습니다.

---

## 2. 구성 요소와 역할
- **RayAsyncEngine** (`pvmcts/search/ray/ray_async.py`): 트리 선택/확장/백업 + 가상 방문 관리. 배치 추론을 비동기로 큐잉·반영.
- **BatchInferenceManager** (`pvmcts/search/ray/batch_inference_manager.py`): 배치 큐/타임아웃/동시 배치 한도 관리, `infer_async` 호출·결과 회수. *(주의: min_batch_size는 없음. 조건은 batch_size 또는 max_wait_ms)*.
- **RayInferenceClient** (`inference/ray_client.py`): Ray actor에 비동기 추론 요청을 던지고 ObjectRef 반환.
- **AlphaZeroAgent & Runners** (`alphazero/agent.py`, `.../runners/`): Runner(SelfPlay/Vectorize/RayWorker)가 `get_action_probs`/`get_action_probs_batch`를 호출하면, Agent 내부 PVMCTS가 RayAsyncEngine을 사용해 정책을 생성합니다.
- **Action 계약**: 전 구간 flat index(int). (x,y) 변환은 Runner에서만 수행.

데이터 흐름: Runner → Agent → PVMCTS(RayAsyncEngine) → BatchInferenceManager → RayInferenceClient → Ray actor(GPU) → 결과 → Engine이 트리에 반영.

### 전체 구조도 (텍스트)
```
[ RayAsyncEngine (CPU) ]        [ BatchInferenceManager ]      [ Ray Client ]      [ Ray Server (GPU) ]
       |                                  |                        |                       |
 1. Selection  ------------------------> Enqueue                   |                       |
       |                                  |                        |                       |
 2. Dispatch? (ready?) ---------------> Dispatch (send) -------> infer_async -----------> (compute)
       |                                  |                        |                       |
 3. Selection (continues)                 (managing inflight)      (ObjectRef)             |
       |                                  |                        |                       |
 4. Drain (any result?) <-------------- Check ready? <--------- ray.wait()                |
       |                                  |                        |                       |
 5. Expansion & Backup <-------------- [Result batch] <------- [Policy, Value] <---------- (done)
```

---

## 3. RayAsyncEngine 루프를 “이해”하며 따라가기
`search(roots, add_noise=False)` 기준, 실제 코드 단계와 개념을 나란히 봅니다.

1) **루트 준비/목표 설정**
   - 단일/리스트 루트를 통일 → `target_visits = params.num_searches`.
   - 역압력 한도 `max_pending = (async_inflight_limit or len(roots)) * batch_size`.
   - 추적용: `pending_by_root`, `inflight_paths`, `inflight_to_root`, `finished_roots`.

2) **Selection & Enqueue** (`_selection_phase`)
   - 종료 조건: `visit_count + pending_by_root[root] >= target_visits` → 더 이상 선택 안 함. pending이 0이면 `finished_roots`에 추가.
   - 역압력: `manager.pending_count() >= max_pending`이면 큐잉을 잠시 중단.
   - 경로 선택: `_select_path` → 리프가 터미널이면 즉시 `backup`, 루트 완료 여부 체크.
   - 비터미널: 경로의 `pending_visits += 1`(가상 방문), `PendingNodeInfo(is_start_node=leaf is root)`와 상태 텐서를 `manager.enqueue`.

3) **Dispatch** (`_dispatch_phase`)
   - `manager.dispatch_ready()`를 반복 호출해 사이즈/타임아웃(`max_wait_ms`) 충족 시 즉시 발송.

4) **Drain/Update** (`_drain_phase`)
   - `manager.drain_ready(timeout_s=0.001)`로 완료 배치를 회수.
   - 매핑된 경로의 `pending_visits`를 0 이상으로 롤백, `pending_by_root` 감소.
   - `_expand_and_backup`: 불법수 마스킹 후 확장, `add_noise=True`면 start node(루트 재사용/첫 확장)에만 Dirichlet 적용, 이후 백업.
   - `target_visits` 도달 + pending 0이면 `finished_roots`에 추가.

5) **정리/예외 처리**
   - 예외/조기 종료 시 inflight 경로의 `pending_visits`를 안전하게 감산, `manager.cleanup()`으로 남은 ObjectRef 취소.

---

## 4. BatchInferenceManager 작동 원리
- `enqueue(mapping, tensor)`: 큐에 적재. `batch_size`를 채웠고 `max_inflight_batches` 한도 이내면 즉시 발송.
- `dispatch_ready(force=False)`: 큐가 있고 inflight 여유가 있으며, 사이즈/타임아웃 조건(`max_wait_ms`)을 만족하면 발송.
- `drain_ready(timeout_s)`: 타임아웃 도달 시 `check_and_flush()`로 강제 발송 → `ray.wait`로 완료 배치 회수 → 완료 시 다음 발송 시도.
- `pending_count()`: 큐 길이 + in-flight 매핑 수 합계.
- `cleanup()`: 남은 ObjectRef 취소, 큐/시계 초기화.

---

## 5. 러너/Agent 연동 (코드와 맞춰 보기)
- **VectorizeRunner** (`alphazero/runners/vectorize_runner.py`): batch self-play에서 `agent.get_action_probs_batch` 호출 → Agent 내부 PVMCTS가 RayAsyncEngine 사용 → 게임 종료 시 `agent.reset_game`으로 슬롯별 루트 리셋.
- **RaySelfPlayWorker** (`alphazero/runners/workers/ray_worker.py`): Ray actor에서 VectorizeRunner를 돌려 다수 게임 생성. `RayAsyncRunner`가 inference actor 생성·가중치 브로드캐스트·worker 배분 담당.
- **SelfPlayRunner** (`alphazero/runners/selfplay.py`): mode="sequential" 등 다른 엔진에서도 동일 Action 계약 유지.
- Config 스케줄링: 러너에서 temperature/dirichlet_epsilon/num_searches 등을 스케줄링해 Agent/PVMCTS에 주입.

---

## 6. 파라미터 해석과 튜닝 포인트
- `batch_size`: 큐가 이 크기를 채우면 즉시 발송. 크면 GPU 효율↑, 지연↑.
- `max_wait_ms`: 부분 배치 타임아웃. 경과 시 발송. 작으면 지연↓, 발송 횟수↑.
- `async_inflight_limit`: 동시 배치 상한. 미지정 시 루트 수만큼 허용(무한 적체 방지). 크면 오래된 priors 위험, 작으면 GPU 놀 수 있음.
- `pending_visits`: 가상 방문수. selection 시 +1, drain/예외 시 반드시 0 이상으로 롤백.
- `add_noise`: 루트 재사용/첫 확장 시에만 Dirichlet 적용(`params.dirichlet_epsilon/alpha`).

---

## 7. 디버깅 체크리스트 (현 코드 기준)
- 목표 방문 초과 방지: `visit_count + pending_by_root` 검사로 over-computation 여부 확인.
- `pending_visits` 음수 방지: 예외 경로에서 롤백이 호출되는지 확인.
- 루트 노이즈: start node에만 적용(깊은 노드에는 적용하지 않음).
- 배치 불일치 방어: `_drain_phase`에서 policy 크기와 매핑 길이가 다르면 RuntimeError 발생.
- 액션 계약: Runner/Agent/TreeNode 모두 flat index만 사용, (x,y) 변환은 Runner에서만 수행.

---

## 8. 확장 아이디어 (개념만 언급)
- C++(pybind11)로 select/backup/virtual loss/비트보드 후보 마스킹/PER priority 갱신을 이식해 Python 루프·GIL 병목을 줄이는 방향 검토.
- Ray 자원 옵션(num_cpus/num_gpus)과 inflight 스케줄을 config에서 주입하도록 runner 스크립트 보완.
