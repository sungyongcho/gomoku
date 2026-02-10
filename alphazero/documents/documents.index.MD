Gomoku AI: from Minimax to Distributed AlphaZero on GCP
I. Minimax‑Based Implementation (Brief Mention)

1.0 Project Goal
학교 프로젝트 요구사항에 맞춰 오목 AI를 구현했습니다. 초기 목표는 과제 평가 기준을 충족하는 것이었으며, 단순한 탐색 기반 AI로도 충분했습니다.

1.1 Python Implementation

1.1.1 Basic Game – 파이썬으로 오목 게임의 규칙과 보드 상태 관리를 구현했습니다.

1.1.2 Capture & Double-Three – 커스텀 룰 구현

1.1.3 Minimax – 순수 Minimax 알고리즘을 적용하여 최적의 수를 탐색했지만, 깊이가 증가할수록 연산 시간이 급격히 늘어났습니다.

1.1.4 python의 한계

배경 – 빠른 개발과 이해를 위해 Python을 선택했으나, 연산 성능이 과제 요구사항을 만족하기에는 부족하다는 점을 확인했습니다.

1.2 Migration to C++

1.2.1 Basic implementation in C++ – Python 구현 이식

1.2.2 Performance Improvement – 체감 가능한 속도 향상

1.3 Optimization Techniques

1.3.1 Alpha‑Beta Pruning – 탐색 공간을 줄이기 위해 알파–베타 가지치기를 적용했습니다. 이는 불필요한 노드를 제외함으로써 계산량을 대폭 감소시켜 성능을 향상시켰습니다.

1.3.2 Quiescence Search – 얕은 깊이에서의 노이즈(급격한 평가 변화) 문제를 완화하기 위해 적용했습니다. 고요상태(quiescent state)가 될 때까지 추가적으로 탐색하여 평가의 안정성을 높였습니다.

1.3.3 PVS (Principal Variation Search) – 가장 유망한 경로를 먼저 탐색하여 분기 수를 줄이는 기법으로, 알파–베타 탐색의 효율을 개선했습니다.

1.4 Project Enhancement

1.4.1 Web Front‑End Integration – 웹 프론트엔드와 연동하여 사용자 인터페이스를 제공하고, AI 플레이를 시각적으로 테스트할 수 있도록 했습니다.

1.4.2 Performance Improvement via Test Cases – 다양한 테스트 케이스를 사용하여 알고리즘의 승률과 실행 속도를 개선했습니다.

1.4.3 Final Evaluation Score – 학교 프로젝트 평가에서 코드 완성도와 성능 측면에서 높은 점수를 획득했습니다.

Transition: From Minimax to AlphaZero

동기 – Minimax 기반 AI는 깊은 수를 탐색할수록 계산량이 기하급수적으로 증가하는 한계가 있습니다. 19×19와 같이 큰 보드에서는 실시간 플레이에 적합하지 않아 학습 기반 접근이 필요했습니다.

한계 – 미리 정의된 휴리스틱에 의존하고, 확장된 규칙(포획·금수)을 모두 반영하기 어렵습니다. 이러한 한계 때문에 강화학습과 MCTS를 결합한 AlphaZero 방식을 도입하게 되었습니다.

II. AlphaZero‑Style Implementation
2.0 Project Goal

19×19 오목(포획 룰과 3‑3 금수 포함)에서 Elo 1800을 달성하는 것을 목표로 합니다. 로컬 개발 환경에서 시작해 GCP 클라우드까지 확장 가능한 분산 학습 시스템을 구축합니다.

2.1 Reinforcement Learning Essentials (Quick Recap)

강화학습의 기본 개념인 정책(policy)과 가치(value) 함수를 간략히 소개하고, 하나의 신경망으로 정책과 가치를 동시에 추정해 MCTS와 결합하는 AlphaZero 방식의 핵심을 설명합니다

2.2 AlphaGo/AlphaZero Overview and Pipeline Structure

자기대국(self‑play)을 통해 데이터를 생성하고, 정책‑가치 네트워크를 업데이트하며, MCTS로 정책을 개선하는 전체 파이프라인을 개괄합니다

2.3 Core Implementation

2.3.1 Gomoku Rules and State Representation – 19×19 오목의 규칙(포획, 3‑3 룰 등)을 설명하고, 게임 상태를 다채널 텐서로 인코딩하는 방식을 소개합니다.

2.3.2 Policy‑Value Network Design – 잔차 블록 기반 CNN 등 아키텍처와 입력 채널 구성, 정책 분포 및 가치 스칼라 출력을 구현하는 방법을 설명합니다.

2.3.3 MCTS & PVMCTS Mechanics – MCTS의 탐색 전략, Dirichlet 노이즈 및 temperature 스케줄링, PVMCTS를 통한 병렬 탐색 구조를 설명합니다.

2.3.4 Evaluation Loop (Challenger vs. Champion, SPRT) – 챌린저와 챔피언 모델 간의 대결, SPRT를 통한 승격 판정, baseline 상대(랜덤·minimax) 및 blunder rate 측정 방법을 기술합니다.

2.3.5 Training Pipeline (Self‑Play → Replay Buffer → Optimization) – 자가대국으로 데이터를 생성하고 리플레이 버퍼에 저장하여 신경망을 최적화하는 과정을 단계별로 설명합니다.

2.4 Distributed Execution Strategies: 단계별 확장

2.4.1 Vectorized Self‑Play & Batched Inference – 단일 프로세스에서 여러 게임을 동시에 실행하는 벡터화 기법과 배치 추론을 활용한 효율적 처리 방식을 설명합니다.

2.4.2 Python Multiprocessing Parallelism – 멀티코어 CPU를 활용해 자기대국과 학습을 병렬화하는 방법과, 발생한 문제점·해결책을 소개합니다.

2.4.3 Ray‑Based Actor Architecture – 여러 노드로 확장할 때 Ray를 사용하는 구조를 설명합니다. 액터 간 통신 방식, 모델 공유 방법, 장애 대응 전략 등을 포함합니다.

2.5 5×5 Experimentation and Hyperparameter Tuning

2.5.1 Local Ray Training Runs on 5×5 Boards – 5×5 보드에서 초기 실험을 수행하며 발견한 버그, 성능 측정, 파라미터 검증 과정을 기록합니다.

2.5.2 Hyperparameter Research (Temperature, Dirichlet, Random Play Ratio, SPRT, etc.) – temperature, Dirichlet 노이즈, 랜덤 플레이 비율, SPRT 임계값 등 하이퍼파라미터를 탐색하고 조정한 결과를 요약합니다.

2.6 Production‑Scale Training on GCP (19×19)

2.6.1 Code Optimization & Calibration for Large‑Scale Ray – 대규모 클러스터에서 발생하는 성능 병목을 해결하기 위해 적용한 최적화 기법과 실험 설정을 설명합니다.

2.6.2 Network Configuration (VPC/NAT, 방화벽, Private IP 워커) – Ray 클러스터를 구성할 때 사용한 VPC, NAT, 방화벽 설정 등 네트워크 구성을 구체적으로 설명합니다.

2.6.3 Cluster Layout: Roles, Resources, and Cost Management – 헤드/워커 노드 역할 분담, 적합한 인스턴스 유형 선택, 예약 인스턴스를 통한 비용 절감 전략을 소개합니다.

2.6.4 Training Results & Monitoring – GCP에서 학습한 결과(Elo, 승격 통계 등)를 정리하고, manifest.json과 로그를 통해 학습 진행 상황을 모니터링한 방법을 설명합니다. 알람 및 로그 수집 도구를 활용해 어떤 지표를 추적했는지 기술합니다.

2.6.5 Tooling & Automation – 대규모 실험을 관리하기 위해 작성한 스크립트(run_batch.py, run_mp.py 등), 설정 스케줄링 도구, 자동 체크포인트 및 재시작 전략을 묶어 소개합니다.

2.7 Lessons & Next Steps

2.7.1 Key Lessons & Pitfalls – 프로젝트 전반에서 발견한 병목, 데이터 품질 이슈, baseline 튜닝 경험, 클러스터 장애 대응 등 핵심 교훈을 정리합니다.

2.7.2 Additional Hyperparameters/Curriculum Ideas – 향후 시도해 볼 추가 하이퍼파라미터 조합이나 커리큘럼 학습 전략을 제안합니다.

2.7.3 Evaluating Against Minimax‑Based Opponents – Minimax 기반 상대와의 비교 평가를 통해 모델의 강점과 약점을 분석합니다.

2.7.4 Future Performance Work – CPython 최적화, JAX 커널, 커스텀 CUDA 파이프라인 등 성능 향상을 위한 연구 방향을 제시합니다.

Conclusion

프로젝트 요약 – Minimax 기반 구현에서 시작해 AlphaZero 스타일의 강화학습 AI로 발전시키는 과정과 분산 학습 시스템 구축까지의 여정을 요약합니다.

성공과 배운 점 – Python 대비 C++로의 성능 개선과 탐색 최적화 기법의 효과, Minimax의 한계를 느끼고 강화학습으로 전환한 과정, GCP 클러스터 운영 경험에서 얻은 교훈 등을 강조합니다.

미래 방향 – AlphaZero 모델을 더욱 효율적으로 개선하기 위한 하이퍼파라미터 연구와 커리큘럼 학습, 클러스터 확장성과 비용 최적화, 그리고 최적화된 라이브러리(JAX, CUDA 등) 도입 가능성을 제시합니다.
