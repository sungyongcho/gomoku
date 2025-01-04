import type { Stone, History } from "~/types/game";
import { range, pipe, toArray, map } from "@fxts/core";

export const useGameStore = defineStore("game", () => {
  const { playStoneSound, playUndoSound } = useSound();
  const { getCapturedStones } = useStoneLogic();
  const settings = ref({
    capture: true,
    doubleThree: true,
    totalPairCaptured: 5,
    firstMove: true,
    advantage1: 0,
    advantage2: 0,
    isPlayer2AI: true,
  });
  const initialBoard = () => {
    return pipe(
      range(19),
      map((_) =>
        pipe(
          range(19),
          map((_) => ({ stoneType: "" as Stone })),
          toArray,
        ),
      ),
      toArray,
    );
  };
  const turn = ref<Stone>("X"); // Player1 = 'X', Player2 = 'O'
  const histories = ref<History[]>([]);
  const boardData = ref<{ stoneType: Stone }[][]>(initialBoard());
  const initGame = () => {
    turn.value = "X";
    histories.value = [];
    boardData.value = initialBoard();
  };

  const changeTurn = () => {
    turn.value = turn.value === "X" ? "O" : "X";
  };
  const historyToLog = (h: History) => {
    const player = h.stoneType === "X" ? "Black" : "White";
    const x = String.fromCharCode("A".charCodeAt(0) + h.coordinate.x);
    const y = h.coordinate.y + 1;

    return `${player} - (${x}, ${y})`;
  };
  const addStoneToBoardData = (
    { x, y }: { x: number; y: number },
    stone: Stone,
  ) => {
    // Check double-three

    // Check if captured stone exist
    const capturedStones = getCapturedStones({
      x,
      y,
      stone,
      boardData: boardData.value,
    });

    // Add to history
    histories.value.push({
      coordinate: { x, y },
      stoneType: stone,
      capturedStones: capturedStones,
    });

    // Update board
    boardData.value[y][x].stoneType = stone;
    capturedStones.forEach(({ x, y }) => {
      boardData.value[y][x].stoneType = "";
    });
    playStoneSound();
    changeTurn();
  };

  const deleteLastHistory = () => {
    if (histories.value.length === 0) return;

    const lastHistory = histories.value[histories.value.length - 1];

    // Recover captured stones
    if (lastHistory.capturedStones) {
      lastHistory.capturedStones.forEach(({ x, y }) => {
        boardData.value[y][x].stoneType =
          lastHistory.stoneType === "X" ? "O" : "X";
      });
    }

    // Undo last move
    boardData.value[lastHistory.coordinate.y][
      lastHistory.coordinate.x
    ].stoneType = "";

    // Delete last history
    histories.value.pop();
    playUndoSound();
    changeTurn();
  };

  const player1TotalCaptured = computed(() => {
    return histories.value
      .filter((h: History) => h.stoneType === "X")
      .reduce((acc: number, h: History) => {
        if (!h.capturedStones?.length) return acc;
        return 1 + acc;
      }, 0);
  });

  const player2TotalCaptured = computed(() => {
    return histories.value
      .filter((h: History) => h.stoneType === "O")
      .reduce((acc: number, h: History) => {
        if (!h.capturedStones?.length) return acc;
        return 1 + acc;
      }, 0);
  });

  return {
    settings,
    turn,
    histories,
    boardData,
    initGame,
    changeTurn,
    historyToLog,
    addStoneToBoardData,
    deleteLastHistory,
    player1TotalCaptured,
    player2TotalCaptured,
  };
});
