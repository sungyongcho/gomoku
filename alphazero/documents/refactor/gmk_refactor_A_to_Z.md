# gmk → gmk‑refactor: 남은 정리 포인트 (현행 코드 기준)

이 문서는 기존 A→Z 가이드에서 **이미 구현된 내용은 걷어내고**, 현재 코드베이스와 어긋나는 미완/정리 필요 작업만 남긴 요약본입니다.

---

## 현재 구현 스냅샷
- 코어/모델/MCTS: `gomoku/core`, `gomoku/model`, `gomoku/pvmcts` 정리 완료.
- Agent·Runner: `gomoku/alphazero/agent.py`, `runners/selfplay.py`, `runners/vectorize_runner.py`, Ray 경로(`runners/ray_runner.py`, `runners/workers/ray_worker.py`) 구현 완료.
- 데이터·학습: `learning/dataset.py`(Parquet 저장·복원, PER 필드 포함), `learning/trainer.py`(AMP, PER 가중치 적용) 동작.
- 평가: `eval/arena.py`, `metrics.py`, `sprt.py` 존재하며 챔피언/베이스라인 승격 조건 로직 포함.
- 스크립트: `gomoku/scripts/train.py`, `scripts/pipelines/run_*.py` 다수 존재하나, 일부는 오래된 import/API 의존성이 남아 있음.

---

## 남은 작업/정리 스텝

### 0) 스크립트 정리(오래된 import/API 의존성 제거) [v]
- 대상: `gomoku/scripts/train.py`, `gomoku/scripts/pipelines/run_*.py` 경로의 실행 스크립트.
- 조치: 현행 alphazero 모듈 경로(`gomoku/alphazero/learning`, `.../runners`, `agent`)와 일치하도록 import/API 갱신, 중복된 오래된 함수 호출 제거.
- 제외: `scripts_old` 디렉토리는 백업본이므로 이번 정리 범위에서 제외.

### 1) 학습 루프 마무리(체크포인트·재개·샤드 로드) [ ]
- Trainer에 `save_latest/load_latest`(optimizer 상태 포함) 연결하고, 재개 시 manifest와 연동.
- 셀프플레이 샤드 로드→배치 학습 경로를 명확히: 저장/로드 함수와 `flatten_game_records` 사용처 일관화.

### 2) 승격 평가 오케스트레이션 [ ]
- `run_loop.py` 등 메인 루프에서 self-play → 학습 → `eval/arena.run_arena` 호출 → best/latest 갱신까지 자동화.
- 평가 로그/SGF·JSON 저장 여부 확정, 블런더/베이스라인 조건 적용 상태를 manifest나 결과로 남기기.

### 3) 파이프라인/스크립트 일원화 [v]
- `scripts/pipelines/run_*.py`의 오래된 모듈 경로를 정리하여 현행 `gomoku/alphazero/learning`, `runners`, `agent` API로 통일 완료.
- Ray 경로를 `runners/ray_runner.py`/`ray_worker.py` 기반으로 정리하고, manifest 업데이트 로직도 공통 함수로 사용.

### 4) Ray 설정 보완 [ ]
- actor 자원 옵션( num_cpus/num_gpus )을 config에서 주입하도록 보완하고, weight broadcast 훅을 표준화.
- async inflight limit/배치 사이즈 스케줄이 `run_games` 호출 시 반영되는지 점검.

### 5) C++ 최적화 착수(빌드 게이트 후, PVMCTS 하이브리드 포함) [ ]
- 현행 RayAsyncEngine은 선택/백업이 Python이라 GIL/for-loop 병목 존재. `core/rules/doublethree.py`의 `renju_cpp` 외엔 C++ 사용 없음.
- 빌드 스모크 테스트 확보 후 다음 순서로 이식:
  1) C++ 코어 확장에 `select_batch`/`backpropagate_batch` 추가(virtual loss/FPU/비트보드 후보 마스킹 포함).
  2) pybind11 바인딩으로 Python에서 select/backup 호출 노출.
  3) Ray 하이브리드: C++ 배치 → Ray 비동기 추론 → 결과를 C++에 주입, 배치 길이/num_searches 방어 로직 유지.
- PER 우선순위 갱신도 대형 버퍼면 C++ SIMD/병렬 갱신 후보에 포함.

---

## 주의할 함정(현행 유지)
- Action 인덱싱 혼용 금지: 전 구간 flat index → (x,y) 변환은 단일 헬퍼만 사용.
- outcome(z) 관점: 턴 플레이어 기준 부호 반전 확인.
- 불법 수 마스킹 후 확률 정규화/NaN 방지.
- Ray 도입 전 단일 스레드/벡터라이즈 경로를 먼저 안정화.
