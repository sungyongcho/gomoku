export type Stone = "O" | "X" | ".";
export type BoardStone = { x: number; y: number; stone: Stone };
export type History = {
  stone: Stone;
  coordinate: { x: number; y: number };
  capturedStones?: BoardStone[];
};

export type BoardInput = {
  x: number;
  y: number;
  stone: Stone;
  boardData: { stone: Stone }[][];
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

export type SocketMoveResponse = {
  type: "move";
  status: "success" | "error";
  board: Stone[][];
  player: Stone;
  capturedStones: { x: number; y: number; stone: Stone }[];
  newStone: { x: number; y: number; stone: Stone };
};
