export type Stone = "O" | "X" | "";
export type History = {
  stoneType: Stone;
  coordinate: { x: number; y: number };
  capturedStones?: { x: number; y: number; stoneType: Stone }[];
};
