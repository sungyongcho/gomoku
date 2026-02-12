gomoku 게임 규칙

---

- 두 명의 플레이어가 19x19 그리드의 교차점에 바둑판 내에서 각기 다른 색(흑/백)의 바둑돌을 착수하고 두가지 조건중 하나를 충족 했을 때 승리하는 게임
  - 5개의 바둑돌을 연속 되게 놓았을 때 (오목)
  - 10개의 상대방 돌을 잡아 먹었을 때 (catch)
  - 무승부
  - 게임을 플레이 하는 도중 특정 조건에 의해 착수를 할 수 없는 위치가 발생 (doublethree)
- 시간 제한 없음

1. 5개의 바둑돌을 연속 되게 놓았을 때 (이부분은 없었(거나 짚고 넘어가지 않은) 부분인데 맨 마지막 항목은 정리가 필요합니다)
   - 플레이어가 5개 혹은 이상이 되는 돌을 연속으로 착수하여 오목이 이루어진 뒤 (게임이 바로 끝나지 않음), 상대 편이 돌을 'catch' 할 수 없는 경우 게임을 승리 하게됨
     - 오목(혹은 이상)을 달성한 상태에서, 그중의 돌이 capture 가 되는 경우 오목이 깨지고 게임은 재진행됨
   - 그러나 상대방에게 주어진 capture의 기회에서 다른 돌을 잡아먹게 되어 10개의 돌을 catch 한 경우
     - 10개의 돌을 catch 했기 때문에 돌을 잡아먹은쪽이 이김
     - 오목이 먼저 달성 되었기 때문에 잡아먹힌 돌과 상관 없이 오목을 둔 쪽이 이김
     - (A player who manages to line up five stones wins only if the opponent cannot break this line by capturing a pair.)
     - 해석이 잘못 되었는지 확인 필요
     - 렌주룰의 경우 오목을 달성하면 바로 승리요건 달성

2. 10개의 돌을 catch 한 경우
   - 한 색깔의 돌이 2개 이어져 있고, 상대방의 돌이 양 끝에 놓아지게 되는 경우 상대방의 돌을 따먹을 수 있음
     - 무조건 2개만, 3개나 이상은 X
   - 상대방의 돌을 10개 (5번) 따먹을 경우 게임 승리
   - 바둑돌을 따먹은 이후, 자신 혹은 상대편은 남아 있는 빈 공간에 돌을 착수 할 수 있음
   - 만약 2개의 돌이 이어져 있지 않은 상태에서 양 끝의 상대방 돌이 놓여 있고, 자신을 돌을 하나씩 놓는 경우는 해당사항이 없음 (subject pdf 10쪽 참고)

3. 위 두 조건을 모두 충족하지 않고, doublethree가 충족된 상태에서 더이상 바둑돌을 둘 수 없는 경우
   - 바둑판이 바둑돌로 꽉 차는 경우
   - 바둑판이 doublethree 규칙 때문에 착수할 수 없는 경우를 제외하고 바둑판이 꽉 차있을 경우

4. doublethree (삼삼)
   - 3개 이상이 이어진 바둑돌 중 한 바둑돌을 기준으로 다른 3개이상의 이어진 바둑돌을 교차하게 만드는 경우
   - 바둑판 모양 기준 직각, 대각선 모두 해당
   - 과제에서 요구하는 삼삼의 룰은 '렌주룰'; '열린 4'를 만들 수 있는 상태를 말함
     - https://namu.wiki/w/%EC%98%A4%EB%AA%A9/%EB%A3%B0%EC%9D%98%20%EC%A2%85%EB%A5%98#s-2.4.1
     - 과제 pdf 그림 참조
   - 과제 pdf 두번째에 나오는 그림의 경우, b 지점을 무시 하고 a 점으로 바로 착수 하려고 하면, 두개의 열린 4를 만들 수 있기 때문에 삼삼으로 착수가 금지되지만, b점에 돌이 미리 존재 한 상태에서는 착수가 가능함. 이는 b 돌이 가로선의 돌의 '열린 4' 를 막아버리기 때문.

- 착수를 통해 돌을 따먹고 이 때문에 doublethree의 모양이 만들어지는 경우, doublethree가 간접적으로 발생하였기 때문에 doublethree 규칙은 해당되지 않음

- 구현시 모든 규칙은 doublethree 발생여부를 우선으로 확인한 이후 나머지 규칙들을 적용하게됨

doublethree → 캡처 → 오목 → 무승부 순으로 규칙 검사

33 확인법

- 착수하는 돌이 끝에 위치
  - .$OO. : 8가지 (단, X.$OO.X, $OO.O 인 경우는 제외해야함)
  - .$O.O.: 8가지 (단, O.$O.O 인 경우 제외)
  - .$.OO.: 8가지
- 착수하는 돌이 중간에 위치
  - .O$O.: 4가지 (단, X.O$O.X, O$O.O 인 경우는 제외해야함)
  - .O$.O.: 8가지

## 12/01/2025

승리조건 체크

- 오목이 되었다 하더라도 하나라도 끊길 수 있으면 게임은 계속 진행.
- 캡쳐 점수가 최대에 달하면 게임 승리
- 오목이 더이상 잡히지 않으면 게임 승리
  edge case
- 6목 달성 후 상대방이 끊어도 5목일때?
- 상대방이 잡아야 하는데 안잡고 방치하면? -> 바로 상대방이 승리

무승부는 front에서 구현

docker comopose 실행법

- `source alias.sh` 실행후 아래 4기지 명령어 실행
  - dev => "docker compose -f docker-compose.yml"
  - dev-up => "docker compose -f docker-compose.yml up"
  - dev-up-build => "docker compose -f docker-compose.yml up --build"
  - dev-down => "docker compose -f docker-compose.yml down"

backend reset example

example_move.json (착수용)

```
{
  "type": "move",
  "new_stone": {
    "x": 1,
    "y": 1,
    "player": "X"
  }
}
```

example_reset.json (리셋용)

```
{
  "type": "reset"
}
```

response_move.json (backend 착수응답)

```
{
  "type": "move",
  "status": "success",
  "board": "...................\n.X.................\n...................\n...................\n...................\n...................\n...................\n...................\n...................\n...................\n...................\n...................\n...................\n...................\n...................\n...................\n...................\n...................\n...................\n"
}
```

response_error.json (backend 착수 에러)

```
{
  "type": "error",
  "error": "Invalid move"
}
```

## 게임 종료 체크 알고리즘

1. 착수를 하기 전
   - 착수를 하기 전에 둘중 아무나 목표점수을 딴 상태라면 게임종료.
   - A턴 이고 착수를 하기 전인데 A가 5목이상이 있다면 승리. (상대방이 바보수를 둘 경우 여기서 걸러짐)
   - 둘 수 있는 공간이 없을 경우 무승부

2. 착수를 한 후에
   - A턴 이고, 착수를 한 후에 완벽한 5목이상이 만들어지면 승리.
   - A턴이고, 착수를 한 후에 불완전한 (breakable) 5목이상이면 게임 계속 진행.

## 3/7/2025

- (front) revert to not placing for doublethree neeeds to be implemented.

## 3/20/25

- (front) needs to check how the check rendering is actually getting handled b/c the capture information on the board is all applied already.

## 3/27/05

- TODO(front)
- doublethree로 착수가 불가능한 경우 마지막 착수가 된 지점의 보드의 돌이 그대로 남아 있는 경우가 있음
- capture를 진행 할 경우, 현재 백엔드에서 보드판과 점수를 모두 업데이트 하는데 프론트에서 가지고 있을 필요가 있는지...?
- 디버그 모드에서 플레이어 자동 변경 토글 버튼이 가능한지?

## 3/27/25 -2

- 33

## 4/4/25

- PV move ordering is implemented but processing time is bigger when depth = 10, but when depth = 8 performance is good and it's under 500ms
- Killer move also implemented, but processing time goes same as pm move ordering, so giving overheads

## 5/24/25

- update endpoint address of minimax
- check connection front <-> minimax

## 5/24/25 - 2

- debug 모드에서 33, capture 안되도록 변경
- /debug 페이지에서는 /ws/debug/
- AI setting 추가 후 api 경로 debug 삭제 (local 은 포트 변경, prod 는 minimax / alphazero)
- footer (`<a href="">sungyong cho</a> (<a>sungyongcho@email.com</a>`)
- production 에서는 debug 페이지 보이지 않게
- nuxt content (/about-project/)
- ws 접속이 안되면 ai 관련 버튼 disabled 알림팝업 띄우기

----------------------NEW--------------------------

# 2026.02.03

- minimax 버전 오픈소스 ForbiddenPointFinder 와 비트 보드 버전 구현, 동작확인
- 프론트 package.json dependency 오류 수정, docker 노드 컨테이너 버전 고정됨 (확인 필요)
- 부탁드림: docs 에서 아이콘을 눌렀을 때 예시 항목을 누른것 펼치도록 아코디언 처리 가능한지?
- 부탁드림: 옆 메뉴 바는 옆 위치에 고정되어 있도록 처리 할 수 있는지?

# 2026.02.10

- 연결 테스트 완료
  - doublethree 프론트엔드/백엔드 로직이 매치가 되지 않아 frontend쪽 doublethree 패치가 필요한 상황
  - evaluaton test 는 minimax 만 적용 하는것으로 명시 가능한지? 현재 alphazero 백엔드 에서도 동작은 하도록 설계 되어 있으나, 모델이 evaluation 케이스에 대응하도록 학습 되어있는지 알 수 없음
- alphazero, minimax 공통 (헷갈려서 질문)
  - undo a move를 하면 ai와 player의 착수 돌의 색깔이 갑자기 뒤바뀌게 됨
  - 게임 플레이는 문제가 없으나 우측 상태바에 player, ai의 착수turn, 점수가 제대로 렌더링 되지 않고 있음
- alphazero 옵션 (고민중 - 아래 부분은 option을 살려 두는게 맞을지, 아니면 option을 살려두고 documentation에서 설명을 하는게 맞을지 의견을 한번 여쭈어 보고 싶습니다)
  - enable capture 삭제: 모델 설계 자체가 capture를 가정하고 학습 된 모델이기 때문에 capture가 꺼진 경우 capture가 없음을 인지하지 못하고 capture수를 둠
  - total captured paire stones 제거: 모델 트레이닝 기본 설정값인 capture: 5를 제공 한 상태에서 학습 하였기 때문에 예상하지 못한 판세에서 모델이 매우 약해짐
  - player advantage 삭제 혹은 매우 낮은 숫자: 모델이 상대방의 score가 높은 경우를 의식 하지 못한 상태에서의 상황에 대해 (특히 점수가 매우 높을 경우) 의 상황이 학습되지 않음
- 참고 사항
  - alphazero 에서는 difficulty가 조절 안 되는 현재 상태 그대로 유지
  - 모델은 따로 catch로 만들어진 오목을 끊어 냈을 경우에 대한 트레이닝을 따로 하지 않았는데 간단하게 게임플레이를 했을때는 잘 막아내고 있음

# 2026.02.11

- alphazero, minimax gcp deploy 기능 완성
- uri 접속 변경 (자세한 내용은 모르지만 cloudflare dns로 설정이 가능했음)
  - 기존 alphazero.sungyongcho.com -> sungyongcho.com/alphazero
  - 기존 minimax.sungyongcho.com -> sungyongcho.com/minimax
  - gomoku.sungyongcho.com 도 cloudflare를 통해서 sungyongcho.com/gomoku로 변경 가능 (계획중)
- TODO:@sucho sungyongcho.com/gomoku 로 프론트엔드 페이지 리다이렉트 기능
  - 프론트엔드 배포가 어디서 되어있는지 확인 필요
- deploy된 alphazero, minimax 의 상태 확인 (docker logs) 은 alias.sh 파일로 컨테이너로 직접 접속하여 확인 가능
- 옵션 토글에서 minimax, alphazero 변경이 제대로 안 일어나는 경우가 있음
  - 토글을 진행하면 connection이 전환되지 않고 유지되고 있음
  - 백엔드 서버에 문제일 수도 있어 확인 필요함
  -> frontend 코드에서 바뀌도록 수정 처리함 (확인 부탁드립니다)

# 2026.02.12

- 인스턴스 타입을 기존 e2-standard-2 (vCPU2) 에서 c2d-standard-4 타입으로 변경 진행
  - 비용이 많이 발생 하기 때문에 약 40일 정도 간격으로 인스턴스를 옮겨 다녀야 함
  - 프론트 페이지에 랜딩을 하기 전에 sungyongcho.com/minimax, sungyongcho.com/alphazero 의 연결 상태를 확인 하고, 응답이 없을 경우 팝업 모달을 띄워인 (maintanance 중) 이라는 메세지를 띄울 수 있을지?

- 간단 설명서
  - 0. .env.example을 .env 로 복사
  - 1. source alias.sh 로 필요한 커맨드들 저장
  - 2. 이후 dev -build 로 필요 이미지 빌드
  - 3. dev-up 으로 전체 front, alphazero, minimax 컨테이너들이나 dev-up-front로 프론트 컨테이너 생성 (dev-up-front 는 FROM_WHERE=prod 일때 유용)
  - 4. 02.11 에서 언급된 deploy 된 alphazero, minimax의 착수 로그 확인
    - deploy-logs-minimax
    - deploy-logs-alphazero
  - alphazero의 성능 (mcts search 횟수) 조절 방법 (로컬에서 개발 시):
    - alphazero/server/websocket.py:28번 줄 NUM_SEARCHES_OVERRIDE: int | None =동None
    - 뒷부분의 None을 수정 (현재 기본값 200, 최대 2400까지 가능 하나 시간이 오래걸림)

- alphazero, minimax vm은 지금부터 이슈가 있지 않는 한 상시 구동 (4~5월 까지 예상)
---------------------------------------

# 부탁 & 의논사항 정리 (위 내용과 중복됨, 정리본)
- 부탁드림: doublethree 프론트엔드/백엔드 로직이 매치가 되지 않아 frontend쪽 doublethree 패치가 필요한 상황
- doublethree 무조건 다 금지
  ```c++
  bool CForbiddenPointFinder::IsDoubleThree(int x, int y)
  ```
- 부탁드림: docs 에서 아이콘을 눌렀을 때 예시 항목을 누른것 펼치도록 아코디언 처리 가능한지?
- 부탁드림: 옆 메뉴 바는 옆 위치에 고정되어 있도록 처리 할 수 있는지?
- 부탁드림: evaluaton test 는 minimax 만 적용 하는것으로 명시 가능한지? 현재 alphazero 백엔드 에서도 동작은 하도록 설계 되어 있으나, 모델이 evaluation 케이스에 대응하도록 학습 되어있는지 알 수 없음
- alphazero, minimax 공통
  - ai와 player 가 둘때는, undo a move를 하면 player 의 차례의 이전수로 돌아가야한다. undo a move -> undo a turn 으로 바꾸자.

의견을 부탁드림:
- alphazero 옵션 option을 살려 두되, default 설정으로 돌아가서 disalbed.
- alphazero 가 선택되면, 난이도 대신 작은 infobox 추가.
-  sungyongcho.com/gomoku 로 프론트엔드 페이지 리다이렉트 기능 (구현 아직 X)
  - 프론트엔드 배포가 어디서 되어있는지 확인 필요
- 부탁드림: 프론트 페이지에  랜딩을 하기 전에 sungyongcho.com/minimax, sungyongcho.com/alphazero 의 연결 상태를 확인 하고, 응답이 없을 경우 팝업 모달을 띄워서 (maintanance 중) 이라는 메세지를 띄울 수 있을지?
- localhost:4000/ws 에 계속 시도하는 버그