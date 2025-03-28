export type Stone = "O" | "X" | ".";
export type BoardStone = { x: number; y: number; stone: Stone };
export type History = {
  stone: Stone;
  coordinate: { x: number; y: number };
  capturedStones?: BoardStone[];
  executionTime?: { s: number; ms: number; ns: number }; // ms
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

export type GameSituation = {
  x: number;
  y: number;
  boardData: { stone: Stone }[][];
  turn: Stone;
  captured: {
    player1: number;
    player2: number;
    goal: number;
  };
};

export type GameResult = {
  result?: GAME_END_SCENARIO;
  winner?: Stone;
};
export type RequestType = "move" | "evaluate";
export type ResponseType = "move" | "evaluate" | "error";
export type SocketMoveRequest = {
  type: RequestType;
  difficulty: "easy" | "hard"; // default hard
  nextPlayer: Stone;
  goal: number;
  lastPlay?: {
    coordinate: { x: number; y: number };
    stone: Stone;
  };
  board: Stone[][];
  scores: { player: Stone; score: number }[];
};

export type SocketDebugMoveResponse = {
  type: ResponseType;
  status: "success" | "doublethree";
  lastPlay: {
    coordinate: { x: number; y: number };
    stone: Stone;
  };
  board: Stone[][];
  capturedStones: { x: number; y: number; stone: Stone }[];
  scores: { player: Stone; score: number }[];
};
