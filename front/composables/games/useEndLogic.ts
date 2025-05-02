import type {
  BoardInput,
  BoardStone,
  GameResult,
  GameSituation,
  Stone,
} from "~/types/game";
import { GAME_END_SCENARIO } from "~/types/game";
import { useBaseLogic } from "~/composables/games/useBaseLogic";
import { useDoubleThreeLogic } from "./useDoubleThreeLogic";

export const useEndLogic = () => {
  const { checkDoubleThree } = useDoubleThreeLogic();
  const { directions, moveToEdgeOfLine, isOutOfBound, getOppositeStone, move } =
    useBaseLogic();

  const getStonesInLine = (
    { x, y, stone, boardData }: BoardInput,
    direction: { dx: number; dy: number },
    res: BoardStone[] = [],
  ): BoardStone[] => {
    if (isOutOfBound({ x, y }) || boardData[y][x].stone !== stone) return res;

    return getStonesInLine(
      { x: x + direction.dx, y: y + direction.dy, stone, boardData },
      direction,
      [...res, { x, y, stone }],
    );
  };

  const isCaptureEnded = (situation: GameSituation): GameResult => {
    const {
      captured: { player1, player2, goal },
    } = situation;

    if (player1 >= goal || player2 >= goal) {
      return {
        result: GAME_END_SCENARIO.PAIR_CAPTURED,
        winner: player1 >= goal ? "X" : "O",
      };
    }
    return {};
  };

  const isPerfectFiveEnded = (situation: GameSituation): GameResult => {
    const { settings } = storeToRefs(useGameStore());
    const winLines: BoardStone[][] = [];
    const { x, y, turn, boardData } = situation;

    for (const { x: dx, y: dy } of directions.slice(0, 4)) {
      const edgeStone = moveToEdgeOfLine(
        { x, y, stone: turn, boardData },
        { dx, dy },
      );

      const stonesInLine = getStonesInLine(
        { x: edgeStone.x, y: edgeStone.y, stone: edgeStone.stone, boardData },
        { dx: -dx, dy: -dy },
      );

      if (stonesInLine.length >= 5) {
        winLines.push(stonesInLine);
      }
    }

    // Filter lines with 5 or more stoneslength >= 5);
    const oppositeStone = getOppositeStone(turn);

    if (winLines.length === 0) return {};

    if (!settings.value.enableCapture) {
      return {
        result: GAME_END_SCENARIO.FIVE_OR_MORE_STONES,
        winner: turn,
      };
    }

    // Check non-breakable line exist
    for (const winLine of winLines) {
      const inclination = {
        dx: winLine[1].x - winLine[0].x,
        dy: winLine[1].y - winLine[0].y,
      };
      const nonParallelDirections = directions
        .filter(({ x, y }) => x !== inclination.dx || y !== inclination.dy)
        .filter(({ x, y }) => x !== -inclination.dx || y !== -inclination.dy);

      for (const stoneOnLine of winLine) {
        for (const { x: _dx, y: _dy } of nonParallelDirections) {
          // Pattern1. _$OX
          // Pattern2. _O$X
          const expected = [".", turn, oppositeStone];
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
                boardData[st.y][st.x].stone === expected[idx],
            ) ||
            stonesPattern2.every(
              (st, idx) =>
                !isOutOfBound(st) &&
                boardData[st.y][st.x].stone === expected[idx],
            )
          ) {
            // Breakable stones exist
            stoneOnLine.stone = ".";
          }
        }
      }
    }

    // Check still 5 stones exist in serial
    for (const winLine of winLines) {
      let cnt = 0;
      for (const stoneOnLine of winLine) {
        if (stoneOnLine.stone === turn) cnt++;
        else if (cnt < 5) cnt = 0;
        else break;
      }
      if (cnt >= 5)
        return {
          result: GAME_END_SCENARIO.FIVE_OR_MORE_STONES,
          winner: turn,
        };
    }

    // Perfect 5 stones not exist
    return {};
  };

  const isDrawEnded = (situation: GameSituation): GameResult => {
    const currentTurn = situation.turn;
    const boardData = situation.boardData;

    for (let y = 0; y < boardData.length; y++) {
      for (let x = 0; x < boardData[y].length; x++) {
        if (boardData[y][x].stone !== ".") continue;

        if (!checkDoubleThree({ x, y, stone: currentTurn, boardData }))
          return {};
      }
    }

    return {
      result: GAME_END_SCENARIO.DRAW,
    };
  };

  return {
    isCaptureEnded,
    isDrawEnded,
    isPerfectFiveEnded,
  };
};
