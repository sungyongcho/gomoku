import type { BoardInput, BoardStone, GameResult, Stone } from "~/types/game";
import { GAME_END_SCENARIO } from "~/types/game";
import { useBaseLogic } from "~/composables/games/useBaseLogic";
import { pipe } from "@fxts/core";

type GameSituation = {
  x: number;
  y: number;
  boardData: { stoneType: Stone }[][];
  turn: Stone;
  captured: {
    player1: number;
    player2: number;
    goal: number;
  };
};
export const useWinLogic = () => {
  const { directions, moveToEdgeOfLine, isOutOfBound, getOppositeStone, move } =
    useBaseLogic();

  const getStonesInLine = (
    { x, y, stone, boardData }: BoardInput,
    direction: { dx: number; dy: number },
    res: BoardStone[] = [],
  ): BoardStone[] => {
    if (isOutOfBound({ x, y }) || boardData[y][x].stoneType !== stone)
      return res;

    return getStonesInLine(
      { x: x + direction.dx, y: y + direction.dy, stone, boardData },
      direction,
      [...res, { x, y, stoneType: stone }],
    );
  };

  const isCaptureEnded = ([situation, gameResult]: [
    GameSituation,
    GameResult,
  ]): [GameSituation, GameResult] => {
    const {
      captured: { player1, player2, goal },
    } = situation;

    if (gameResult.result) return [situation, gameResult];
    if (player1 >= goal || player2 >= goal) {
      return [
        situation,
        {
          result: GAME_END_SCENARIO.PAIR_CAPTURED,
          winner: player1 >= goal ? "X" : "O",
        },
      ];
    }
    return [situation, {}];
  };

  const isFiveOrMoreEnded =
    (checkBreakable: boolean) =>
    ([situation, gameResult]: [GameSituation, GameResult]): [
      GameSituation,
      GameResult,
    ] => {
      if (gameResult.result) return [situation, gameResult];
      const lines: BoardStone[][] = [];
      const {
        x,
        y,
        turn,
        boardData,
        captured: { player1, player2, goal },
      } = situation;

      for (const { x: dx, y: dy } of directions.slice(0, 4)) {
        const edgeStone = moveToEdgeOfLine(
          { x, y, stone: turn, boardData },
          { dx, dy },
        );

        const stonesInLine = getStonesInLine(
          { x: edgeStone.x, y: edgeStone.y, stone: edgeStone.stone, boardData },
          { dx: -dx, dy: -dy },
        );
        lines.push(stonesInLine);
      }

      // Filter lines with 5 or more stones
      const winLines = lines.filter((line) => line.length >= 5);
      const oppositeStone = getOppositeStone(turn);
      const oppositeCapturedStones = oppositeStone === "X" ? player1 : player2;

      if (winLines.length > 0 && oppositeCapturedStones < goal - 1) {
        return [
          situation,
          {
            result: GAME_END_SCENARIO.FIVE_OR_MORE_STONES,
            winner: turn,
          },
        ];
      }

      // Check if any stones on win lines can be captured by opponent
      const nonBreakableWinLines = [];
      for (const winLine of winLines) {
        const inclination = {
          dx: winLine[1].x - winLine[0].x,
          dy: winLine[1].y - winLine[0].y,
        };
        const nonParallelDirections = directions
          .filter(({ x, y }) => x !== inclination.dx && y !== inclination.dy)
          .filter(({ x, y }) => x !== -inclination.dx && y !== -inclination.dy);

        let isBreakable = false;
        for (const stoneOnLine of winLine) {
          for (const { x: _dx, y: _dy } of nonParallelDirections) {
            // Pattern1. _$OX
            // Pattern2. _O$X
            const expected = ["", turn, oppositeStone];
            const stonesPattern1 = [
              move(stoneOnLine, { dx: _dx, dy: _dy }, -1),
              move(stoneOnLine, { dx: _dx, dy: _dy }, 1),
              move(stoneOnLine, { dx: _dx, dy: _dy }, 2),
            ];
            const stonesPattern2 = [
              move(stoneOnLine, { dx: _dx, dy: _dy }, -2),
              move(stoneOnLine, { dx: _dx, dy: _dy }, -1),
              move(stoneOnLine, { dx: _dx, dy: _dy }, 1),
            ];
            if (
              stonesPattern1.every(
                (st, idx) =>
                  !isOutOfBound(st) &&
                  boardData[st.y][st.x].stoneType === expected[idx],
              ) ||
              stonesPattern2.every(
                (st, idx) =>
                  !isOutOfBound(st) &&
                  boardData[st.y][st.x].stoneType === expected[idx],
              )
            ) {
              isBreakable = true;
              break;
            }
          }

          if (isBreakable) break;
        }

        if (!isBreakable) {
          nonBreakableWinLines.push(winLine);
          break;
        }
      }

      if (
        (checkBreakable && nonBreakableWinLines.length > 0) ||
        (!checkBreakable && winLines.length > 0)
      ) {
        return [
          situation,
          {
            result: GAME_END_SCENARIO.FIVE_OR_MORE_STONES,
            winner: turn,
          },
        ];
      }

      return [situation, {}];
    };

  const getResult = ([_, result]: [GameSituation, GameResult]) => result;

  const checkWinCondition = (
    gameSituation: GameSituation,
    checkBreakable: boolean = true,
  ): GameResult => {
    return pipe(
      [gameSituation, {}] as [GameSituation, GameResult],
      isCaptureEnded,
      isFiveOrMoreEnded(checkBreakable),
      getResult,
    );
  };

  return {
    checkWinCondition,
  };
};
