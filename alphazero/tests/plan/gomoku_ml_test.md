Me/Opp/Empty 상호 배타성 v
- 모든 좌표에서 Me + Opp + Empty == 1인지 확인.

Last Move 채널 v
- 마지막 수가 있을 때 정확한 위치만 1, 없으면 전부 0인지 검증.

Capture Plane 기본 동작 v
- 포획 점수(p1_pts/p2_pts)를 비율로 브로드캐스트한 값이 정확히 들어가는지 확인.

Color Plane v
- 현재 차례가 흑이면 +1, 백이면 -1로 채워지는지 검사.

Forbidden Plane 좌표 매핑 v
- 금수 패턴을 설정해 두었을 때 정확한 (y, x) 위치에 1이 찍히는지, 축 뒤집힘이 없는지 확인.

Forbidden Plane 활성/비활성 v
-금수 규칙이 켜져 있을 때만 금수 좌표가 1, 꺼져 있으면 항상 0인지 확인.

History Plane 기본 동작 v
- history_length만큼 최근 수가 각 채널에 올바르게 표시되는지, 부족하면 0인지 확인.

History Plane 시간 이동(Temporal Shift) v
- 시점 $T$와 $T+1$을 비교해, history[k+1]가 이전 시점의 history[k]와 정확히 일치하는지 검증.

포획 잔상 (History-Capture Interaction)
- 포획으로 보드에서 사라진 돌이 과거 히스토리 평면에는 남아 있는지 확인. v
- 연속 포획/자충수 상황에서도 히스토리 큐가 꼬이지 않는지 테스트. v

포획 평면 정규화/클리핑
- 목표 점수보다 더 많은 포획을 얻었을 때 값이 1.0을 넘지 않게 클리핑되는지(또는 넘는 값을 허용할지) 정책을 정하고 검증. v
- capture_goal = 0일 때 ZeroDivisionError나 NaN이 발생하지 않고 0으로 채워지는지 확인. v

Out-of-bounds/패딩 영역
-텐서가 보드보다 큰 크기일 경우(패딩 사용) 패딩 영역이 항상 0인지 검증. v

값 범위 검증
- 금수/히스토리/Me/Opp 등 이진 평면이 0.0 또는 1.0만 갖는지 확인 (부동소수 오차 허용). v

결정론성 (Determinism)
-  동일한 GameState로 get_encoded_state를 여러 번 호출해도 결과가 비트 단위로 동일한지 확인. v


배치 간 데이터 오염
- 서로 다른 상태(예: 빈 보드, 가득 찬 보드)를 한 배치에 넣어도 채널 값이 서로 섞이지 않는지 확인. v

게임 종료 직후
- 승리 직후 상태에서도 Last Move, Color Plane 등이 일관된 값을 유지하는지 확인.

가변 보드 크기 대응
- num_lines를 9, 15 등으로 바꿨을 때 출력 텐서 shape이 (B, C, board_size, board_size)로 정확히 바뀌는지 확인.
