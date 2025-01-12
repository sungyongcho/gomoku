import type { BoardInput, BoardStone } from "~/types/game";
import { useBaseLogic } from "~/composables/games/useBaseLogic";

/*
 ** Possible to capture pair stones
 ** ex) XOOX
 */
export const useCaptureLogic = () => {
  const { directions, getOppositeStone, isOutOfBound } = useBaseLogic();

  const getCapturedStones = ({
    x,
    y,
    stone,
    boardData,
  }: BoardInput): BoardStone[] => {
    const finalRes: BoardStone[] = [];
    const oppositeStone = getOppositeStone(stone);
    const expectedPatterns = [oppositeStone, oppositeStone, stone];

    const DFS = (
      coordinate: { x: number; y: number },
      direction: { dx: number; dy: number },
      depth: number,
      capturedStones: BoardStone[],
    ) => {
      if (depth > 2) return capturedStones;

      const [x, y] = [coordinate.x + direction.dx, coordinate.y + direction.dy];
      if (isOutOfBound({ x, y })) return [];

      const stoneInBoard = boardData[y][x].stone;
      if (stoneInBoard !== expectedPatterns[depth]) {
        return [];
      }

      if (depth === 2) {
        // We are not capturing the same stone
        return DFS({ x, y }, direction, depth + 1, capturedStones);
      }

      const newCapturedStones = [
        ...capturedStones,
        {
          x,
          y,
          stone: stoneInBoard,
        },
      ];
      return DFS({ x, y }, direction, depth + 1, newCapturedStones);
    };

    directions.forEach(({ x: dx, y: dy }) => {
      finalRes.push(...DFS({ x, y }, { dx, dy }, 0, []));
    });

    return finalRes;
  };

  return {
    getCapturedStones,
  };
};
