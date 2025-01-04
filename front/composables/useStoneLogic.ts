import type { CapturedStone, Stone } from "~/types/game";

type BoardInput = {
  x: number;
  y: number;
  stone: Stone;
  boardData: { stoneType: Stone }[][];
};
const directions = [
  { x: 1, y: 0 },
  { x: 1, y: -1 },
  { x: -1, y: -1 },
  { x: -1, y: 0 },
  { x: 0, y: -1 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
  { x: -1, y: 1 },
];

const getOppositeStone = (stone: Stone) => {
  if (!stone) return "";
  return stone === "X" ? "O" : "X";
};
const isOutOfBound = ({ x, y }: { x: number; y: number }) => {
  return x < 0 || x > 18 || y < 0 || y > 18;
};
const isOutOfBoundOrOpposite = (
  { x, y }: { x: number; y: number },
  boardData: { stoneType: Stone }[][],
  oppositeStone: Stone,
) => {
  return isOutOfBound({ x, y }) || boardData[y][x].stoneType === oppositeStone;
};

const getCapturedStones = ({
  x,
  y,
  stone,
  boardData,
}: BoardInput): CapturedStone[] => {
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

const move = (
  { x, y }: { x: number; y: number },
  direction: { dx: number; dy: number },
  times: number,
) => {
  return { x: x + direction.dx * times, y: y + direction.dy * times };
};

const is_$OO_ = (
  { x, y, stone, boardData }: BoardInput,
  direction: { dx: number; dy: number },
) => {
  // Handle exception case: X_$OO_X
  const oppositeStone = getOppositeStone(stone);
  const edgeStones = [
    move({ x, y }, direction, -2),
    move({ x, y }, direction, 4),
  ];
  if (
    edgeStones.every((st) =>
      isOutOfBoundOrOpposite(st, boardData, oppositeStone),
    )
  ) {
    return false;
  }

  // Handle _$00_
  const stones = [
    move({ x, y }, direction, -1),
    move({ x, y }, direction, 1),
    move({ x, y }, direction, 2),
    move({ x, y }, direction, 3),
  ];
  const expected = ["", stone, stone, ""];
  if (
    stones.every(
      (st, i) =>
        !isOutOfBound(st) && boardData[st.y][st.x].stoneType === expected[i],
    )
  ) {
    return true;
  }

  return false;
};

const is_$_OO_ = (
  { x, y, stone, boardData }: BoardInput,
  direction: { dx: number; dy: number },
) => {
  const stones = [
    move({ x, y }, direction, -1),
    move({ x, y }, direction, 1),
    move({ x, y }, direction, 2),
    move({ x, y }, direction, 3),
    move({ x, y }, direction, 4),
  ];
  const expected = ["", "", stone, stone, ""];
  if (
    stones.every(
      (st, i) =>
        !isOutOfBound(st) && boardData[st.y][st.x].stoneType === expected[i],
    )
  ) {
    return true;
  }

  return false;
};
const is_$O_O_ = (
  { x, y, stone, boardData }: BoardInput,
  direction: { dx: number; dy: number },
) => {
  const stones = [
    move({ x, y }, direction, -1),
    move({ x, y }, direction, 1),
    move({ x, y }, direction, 2),
    move({ x, y }, direction, 3),
    move({ x, y }, direction, 4),
  ];
  const expected = ["", stone, "", stone, ""];
  if (
    stones.every(
      (st, i) =>
        !isOutOfBound(st) && boardData[st.y][st.x].stoneType === expected[i],
    )
  ) {
    return true;
  }
  return false;
};

const checkEdgeOpenThree = ({ x, y, stone, boardData }: BoardInput) => {
  let count = 0;

  // Pattern1: _$OO_, 8 cases except X_$OO_X
  for (const { x: dx, y: dy } of directions) {
    if (is_$OO_({ x, y, stone, boardData }, { dx, dy })) {
      count++;
    }
  }

  // Pattern2: _$_OO_, 8 cases
  for (const { x: dx, y: dy } of directions) {
    if (is_$_OO_({ x, y, stone, boardData }, { dx, dy })) {
      count++;
    }
  }

  // Pattern3: _$O_O_, 8 cases
  for (const { x: dx, y: dy } of directions) {
    if (is_$O_O_({ x, y, stone, boardData }, { dx, dy })) {
      count++;
    }
  }
  return count;
};

const is_O$O_ = (
  { x, y, stone, boardData }: BoardInput,
  direction: { dx: number; dy: number },
) => {
  // Handle exception case: X_O$O_X
  const oppositeStone = getOppositeStone(stone);
  const edgeStones = [
    move({ x, y }, direction, -3),
    move({ x, y }, direction, 3),
  ];
  if (
    edgeStones.every((st) =>
      isOutOfBoundOrOpposite(st, boardData, oppositeStone),
    )
  ) {
    return false;
  }

  // Handle _$00_
  const stones = [
    move({ x, y }, direction, -2),
    move({ x, y }, direction, -1),
    move({ x, y }, direction, 1),
    move({ x, y }, direction, 2),
  ];
  const expected = ["", stone, stone, ""];
  if (
    stones.every(
      (st, i) =>
        !isOutOfBound(st) && boardData[st.y][st.x].stoneType === expected[i],
    )
  ) {
    return true;
  }

  return false;
};

const is_O$_O_ = (
  { x, y, stone, boardData }: BoardInput,
  direction: { dx: number; dy: number },
) => {
  const stones = [
    move({ x, y }, direction, -2),
    move({ x, y }, direction, -1),
    move({ x, y }, direction, 1),
    move({ x, y }, direction, 2),
    move({ x, y }, direction, 3),
  ];
  const expected = ["", stone, "", stone, ""];
  if (
    stones.every(
      (st, i) =>
        !isOutOfBound(st) && boardData[st.y][st.x].stoneType === expected[i],
    )
  ) {
    return true;
  }
  return false;
};

const checkMiddleOpenThree = ({ x, y, stone, boardData }: BoardInput) => {
  let count = 0;

  // Pattern1: _O$O_, 4 cases except X_O$O_X
  for (const { x: dx, y: dy } of directions.slice(0, 4)) {
    if (is_O$O_({ x, y, stone, boardData }, { dx, dy })) {
      count++;
    }
  }
  // Pattern2: _O$_O_, 8 cases
  for (const { x: dx, y: dy } of directions) {
    if (is_O$_O_({ x, y, stone, boardData }, { dx, dy })) {
      count++;
    }
  }
  return count;
};
const checkDoubleThree = ({ x, y, stone, boardData }: BoardInput) => {
  let count = 0; // if more than 2, it is double three
  count += checkEdgeOpenThree({ x, y, stone, boardData });
  count += checkMiddleOpenThree({ x, y, stone, boardData });
  console.log(count);
  return count >= 2;
};

export const useStoneLogic = () => {
  return {
    getCapturedStones,
    checkDoubleThree,
  };
};
