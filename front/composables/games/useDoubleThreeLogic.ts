import type { BoardInput } from "~/types/game";
import { useBaseLogic } from "~/composables/games/useBaseLogic";

export const useDoubleThreeLogic = () => {
  const {
    directions,
    move,
    isOutOfBound,
    isOutOfBoundOrOpposite,
    getOppositeStone,
  } = useBaseLogic();

  const is_$OO_ = (
    { x, y, stone, boardData }: BoardInput,
    direction: { dx: number; dy: number },
  ) => {
    // Handle exception case: X_$OO_X
    // Handle exception case: $OO_O
    const oppositeStone = getOppositeStone(stone);
    const edgeStones = [
      move({ x, y }, direction, -2),
      move({ x, y }, direction, 4),
    ];
    const fourStones = [
      move({ x, y }, direction, 1),
      move({ x, y }, direction, 2),
      move({ x, y }, direction, 3),
      move({ x, y }, direction, 4),
    ];
    const expectedFourStones = [stone, stone, "", stone];
    if (
      edgeStones.every((st) =>
        isOutOfBoundOrOpposite(st, boardData, oppositeStone),
      ) ||
      fourStones.every(
        (st, i) =>
          !isOutOfBound(st) &&
          boardData[st.y][st.x].stoneType === expectedFourStones[i],
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
    // Handle exception case: O_$O_O
    const exceptionStones = [
      move({ x, y }, direction, -2),
      move({ x, y }, direction, -1),
      move({ x, y }, direction, 1),
      move({ x, y }, direction, 2),
      move({ x, y }, direction, 3),
    ];
    const exceptionExpected = [stone, "", stone, "", stone];
    if (
      exceptionStones.every(
        (st, i) =>
          !isOutOfBound(st) &&
          boardData[st.y][st.x].stoneType === exceptionExpected[i],
      )
    ) {
      return false;
    }

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
    // Handle exception case: O$O_O
    const oppositeStone = getOppositeStone(stone);
    const edgeStones = [
      move({ x, y }, direction, -3),
      move({ x, y }, direction, 3),
    ];
    const fourStones = [
      move({ x, y }, direction, -1),
      move({ x, y }, direction, 1),
      move({ x, y }, direction, 2),
      move({ x, y }, direction, 3),
    ];
    const expectedFourStones = [stone, stone, "", stone];
    if (
      edgeStones.every((st) =>
        isOutOfBoundOrOpposite(st, boardData, oppositeStone),
      ) ||
      fourStones.every(
        (st, i) =>
          !isOutOfBound(st) &&
          boardData[st.y][st.x].stoneType === expectedFourStones[i],
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
    return count >= 2;
  };

  return {
    checkDoubleThree,
  };
};
