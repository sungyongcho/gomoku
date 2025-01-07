export type Stone = "O" | "X" | "";
export type BoardStone = { x: number; y: number; stoneType: Stone };
export type History = {
  stoneType: Stone;
  coordinate: { x: number; y: number };
  capturedStones?: BoardStone[];
};

export type BoardInput = {
  x: number;
  y: number;
  stone: Stone;
  boardData: { stoneType: Stone }[][];
};

export enum GAME_END_SCENARIO {
  FIVE_OR_MORE_STONES = "FIVE_OR_MORE_STONES",
  PAIR_CAPTURED = "PAIR_CAPTURED",
  DRAW = "DRAW",
}

export type GameResult = {
  result?: GAME_END_SCENARIO;
  winner?: Stone;
};
