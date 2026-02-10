# Day6 Recover 분석 및 운영안 (커스텀 룰: 흑/백 더블쓰리 공통)

작성일: 2026-02-07

## 1) 규칙 정합화 결론

- 커스텀 룰 기준: 더블쓰리 금수는 흑/백 모두 적용.
- 기존 구현 문제:
  - `cpp/src/GomokuCore.cpp`는 더블쓰리를 흑 기준으로만 처리.
  - `cpp/src/bindings.cpp`의 `detect_doublethree`는 `player` 인자를 무시해 Python 경로도 실질적으로 흑 기준.
- 수정 방안:
  - 백 차례에서는 보드 색상을 반전(white<->black)해 동일 금수 로직으로 평가.

## 2) "같은 run/replay 유지" 재검증

### 결론

- **같은 run_id 유지 + replay 전량 폐기 후 재수집**을 기본안으로 사용.
- 같은 run + 같은 replay 유지(재사용)는 비권장.

### 로컬 replay 샘플 재측정(white turn, white stones>=4)

- 데이터: `runs/elo1800-gcp-v4/replay/*.parquet`
- 샘플 방식: 파케이 순회 중 일정 간격 샘플링(2026-02-07 측정)
- white 금수 판정은 백 기준 색상 반전 후 금수 탐지로 계산

- 전체 샘플:
  - samples: 4500
  - affected_rate(white forbidden 존재): **1.80%**
  - forbidden_mass_mean(affected only): **0.33%**
  - top1_forbidden_rate(affected only): **0.00%**

- Day3~5 구간은 오염도가 상대적으로 높게 관측:
  - day3: affected 7.22%, forbidden_mass_mean 10.45%, top1_forbidden 10.77%
  - day4: affected 10.14%, forbidden_mass_mean 8.37%, top1_forbidden 8.45%
  - day5: affected 11.11%, forbidden_mass_mean 5.04%, top1_forbidden 4.00%

해석:

- 후반 데이터(day3~5)일수록 새 룰 기준 정책 오염 신호가 증가.
- 규칙 전환 직후 기존 replay를 재사용하면 타깃 불일치가 누적될 수 있음.

## 3) 운영 선택지

1. 권장: 같은 run 유지 + replay 폐기 후 재수집

- 유지: `ckpt/`, `manifest.json`
- 폐기: `replay/*.parquet`

2. 비권장: 같은 run + replay 그대로 유지

- 규칙 변경 이전 샘플과 이후 샘플이 혼재되어 학습 안정성 저하 가능성 큼.

3. 보수안: run_id 신규 생성

- 가장 깨끗하지만 비용/시간 증가.

## 4) Day6 Recover 실행 기준

- 설정 파일: `configs/elo1800-v4-day6-recover.yaml`
- 핵심값:
  - `training.opponent_rates`: random 0.08 / prev 0.25
  - `training.temperature`: 1.10 -> 0.95
  - `mcts.dirichlet_epsilon`: 0.42 -> 0.35
  - `mcts.exploration_turns`: 28
  - `mcts.num_searches`: 1600 -> 2200
  - `training.replay_buffer_size`: 450000
  - `evaluation.eval_opening_turns`: 6
  - `evaluation.eval_temperature`: 1.0
  - `evaluation.eval_dirichlet_epsilon`: 0.2
  - `evaluation.num_baseline_games`: 16
  - `evaluation.baseline_num_searches`: 600

## 5) replay 폐기 실행 예시 (수동)

주의: 아래 명령은 실제 삭제를 수행한다.

```bash
# Local
rm -f runs/elo1800-gcp-v4/replay/*.parquet

# GCS (run_prefix=gmk-testing-refactoring)
gsutil -m rm gs://gmk-testing-refactoring/elo1800-gcp-v4/replay/*.parquet
```

## 6) 가드레일

- Iter+8 시점: `cross4_share(66/174/186/294) > 70%` 이면 탐색/탐험 재상향.
- Iter+16 시점: `center5_share < 4%` 이면 day6-recover 설정 롤백 또는 opponent mix 상향.
- 매 iter 모니터링: `cross4_share`, `center5_share`, `top1_prob`, `nonzero_actions`.
