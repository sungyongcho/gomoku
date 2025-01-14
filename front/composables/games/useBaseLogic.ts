import type { Stone, BoardInput } from "~/types/game";

export const useBaseLogic = () => {
  const getOppositeStone = (stone: Stone) => {
    if (!stone) return ".";
    return stone === "X" ? "O" : "X";
  };
  const isOutOfBound = ({ x, y }: { x: number; y: number }) => {
    return x < 0 || x > 18 || y < 0 || y > 18;
  };
  const isOutOfBoundOrOpposite = (
    { x, y }: { x: number; y: number },
    boardData: { stone: Stone }[][],
    oppositeStone: Stone,
  ) => {
    return isOutOfBound({ x, y }) || boardData[y][x].stone === oppositeStone;
  };
  const directions = [
    { x: 1, y: 0 },
    { x: 1, y: -1 },
    { x: 0, y: -1 },
    { x: -1, y: -1 },
    { x: -1, y: 0 },
    { x: 1, y: 1 },
    { x: 0, y: 1 },
    { x: -1, y: 1 },
  ];

  const move = (
    { x, y }: { x: number; y: number },
    direction: { dx: number; dy: number },
    times: number,
  ) => {
    return { x: x + direction.dx * times, y: y + direction.dy * times };
  };

  const moveToEdgeOfLine = (
    { x, y, stone, boardData }: BoardInput,
    direction: { dx: number; dy: number },
  ) => {
    const moved = move({ x, y }, direction, 1);
    if (isOutOfBound(moved) || boardData[moved.y][moved.x].stone !== stone) {
      return { x, y, stone };
    }
    return moveToEdgeOfLine(
      { x: moved.x, y: moved.y, stone, boardData },
      direction,
    );
  };

  return {
    directions,
    getOppositeStone,
    isOutOfBound,
    isOutOfBoundOrOpposite,
    move,
    moveToEdgeOfLine,
  };
};
