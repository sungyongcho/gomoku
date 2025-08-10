import numpy as np
from gomoku import GameState, Gomoku
from policy_value_net import PolicyValueNet  # 챔피언 모델 로딩을 위해 필요
from pvmcts import PVMCTS
from tqdm import trange


class Arena:
    def __init__(self, game: Gomoku, args: dict):
        self.game = game
        self.args = args

    def _play_a_game(self, p1_model: PolicyValueNet, p2_model: PolicyValueNet) -> int:
        """
        두 모델로 한 판의 게임을 진행하고, P1의 관점에서 결과를 반환합니다.
        (평가 시에는 탐색의 무작위성을 제거하기 위해 argmax를 사용합니다.)

        Returns:
            int: P1의 관점에서 본 게임 결과. 1은 승리, -1은 패배, 0은 무승부.
        """
        p1_mcts = PVMCTS(self.game, self.args, p1_model)
        p2_mcts = PVMCTS(self.game, self.args, p2_model)

        players = [p1_mcts, p2_mcts]
        state: GameState = self.game.get_initial_state()
        turn = 0

        while True:
            current_player_mcts = players[turn % 2]

            # 평가 시에는 항상 가장 좋은 수를 선택 (Greedy)
            action_probs = current_player_mcts.search(state)
            best_action_idx = np.argmax(action_probs)
            action = (
                best_action_idx % self.game.col_count,
                best_action_idx // self.game.col_count,
            )

            state = self.game.get_next_state(state, action, state.next_player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                if value == 1:
                    # 수를 둔 플레이어(current_player)가 이김
                    # P1이 이겼으면 +1, P2가 이겼으면 -1 반환
                    return 1 if current_player_mcts == p1_mcts else -1
                else:
                    return 0  # 무승부

            turn += 1

    def evaluate(self, challenger: PolicyValueNet, champion: PolicyValueNet) -> float:
        """
        지정된 횟수만큼 게임을 진행하여 챌린저의 승률을 계산합니다.

        Returns:
            float: 챌린저의 승률 (0.0 ~ 1.0)
        """
        num_games = self.args["num_eval_games"]
        challenger_wins = 0

        # 절반은 챌린저가 선공(P1)
        for _ in trange(num_games // 2, desc="Evaluating (Challenger as P1)"):
            result = self._play_a_game(challenger, champion)
            if result == 1:  # P1(챌린저) 승리
                challenger_wins += 1

        # 나머지 절반은 챔피언이 선공(P1)
        for _ in trange(num_games // 2, desc="Evaluating (Challenger as P2)"):
            result = self._play_a_game(champion, challenger)
            if result == -1:  # P1(챔피언)이 졌을 때 -> P2(챌린저) 승리
                challenger_wins += 1

        return challenger_wins / num_games
