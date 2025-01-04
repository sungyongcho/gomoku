export type Stone = "O" | "X" | "";
export type CapturedStone = { x: number; y: number; stoneType: Stone };
export type History = {
  stoneType: Stone;
  coordinate: { x: number; y: number };
  capturedStones?: CapturedStone[];
};
