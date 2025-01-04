import type { CapturedStone, Stone } from "~/types/game";

const directions = [
  { x: 1, y: 0 },
  { x: -1, y: 0 },
  { x: 0, y: 1 },
  { x: 0, y: -1 },
  { x: 1, y: 1 },
  { x: 1, y: -1 },
  { x: -1, y: 1 },
  { x: -1, y: -1 },
];

export const useStoneLogic = () => {
  const getOppositeStone = (stone: Stone) => {
    if (!stone) return "";
    return stone === "X" ? "O" : "X";
  };
  const isOutOfBound = ({ x, y }: { x: number; y: number }) => {
    return x < 0 || x > 18 || y < 0 || y > 18;
  };
  const getCapturedStones = ({
    x,
    y,
    stone,
    boardData,
  }: {
    x: number;
    y: number;
    stone: Stone;
    boardData: { stoneType: Stone }[][];
  }): CapturedStone[] => {
    const finalRes: CapturedStone[] = [];
    const oppositeStone = getOppositeStone(stone);
    const expectedPatterns = [oppositeStone, oppositeStone, stone];

    const DFS = (
      coordinate: { x: number; y: number },
      direction: { dx: number; dy: number },
      depth: number,
      capturedStones: CapturedStone[],
    ) => {
      if (depth > 2) return capturedStones;

      const [x, y] = [coordinate.x + direction.dx, coordinate.y + direction.dy];
      if (isOutOfBound({ x, y })) return [];

      const stoneInBoard = boardData[y][x].stoneType;
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
          stoneType: stoneInBoard,
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
