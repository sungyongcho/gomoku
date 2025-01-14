import {
  type Stone,
  type History,
  type BoardInput,
  type BoardStone,
  GAME_END_SCENARIO,
} from "~/types/game";
import { range, pipe, toArray, map } from "@fxts/core";
import { useStorage } from "@vueuse/core";

export const useGameStore = defineStore("game", () => {
  const { doAlert } = useAlertStore();
  const { playStoneSound, playUndoSound } = useSound();
  const {
    getCapturedStones,
    checkDoubleThree,
    isCaptureEnded,
    isCurrentTurnFiveEnded,
    isDrawEnded,
    isPerfectFiveEnded,
  } = useGameLogic();
  const settings = useStorage("settings", {
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
          map((_) => ({ stone: "." as Stone })),
          toArray,
        ),
      ),
      toArray,
    );
  };
  const turn = useStorage<Stone>("turn", "X"); // Player1 = 'X', Player2 = 'O'
  const histories = useStorage<History[]>("histories", []);
  const gameOver = useStorage<boolean>("gameOver", false);
  const boardData = useStorage<{ stone: Stone }[][]>(
    "boardData",
    initialBoard(),
  );
  const player1TotalCaptured = computed(() => {
    return (
      histories.value
        .filter((h: History) => h.stone === "X")
        .reduce((acc: number, h: History) => {
          if (!h.capturedStones?.length) return acc;
          return h.capturedStones?.length / 2 + acc;
        }, 0) + settings.value.advantage1
    );
  });

  const player2TotalCaptured = computed(() => {
    return (
      histories.value
        .filter((h: History) => h.stone === "O")
        .reduce((acc: number, h: History) => {
          if (!h.capturedStones?.length) return acc;
          return h.capturedStones?.length / 2 + acc;
        }, 0) + settings.value.advantage2
    );
  });

  const initGame = () => {
    turn.value = "X";
    gameOver.value = false;
    histories.value = [];
    boardData.value = initialBoard();
  };

  const changeTurn = (t?: Stone) => {
    if (t) turn.value = t;
    else turn.value = turn.value === "X" ? "O" : "X";
  };
  const historyToLog = (h: History) => {
    const player = h.stone === "X" ? "Black" : "White";
    const x = String.fromCharCode("A".charCodeAt(0) + h.coordinate.x);
    const y = h.coordinate.y + 1;

    return `${player} - (${x}, ${y})`;
  };

  const updateBoard = (
    { x, y, boardData, stone }: BoardInput,
    capturedStones: BoardStone[],
  ) => {
    boardData[y][x].stone = stone;
    capturedStones.forEach(({ x, y }) => {
      boardData[y][x].stone = ".";
    });
  };

  const debugAddStoneToBoardData = (
    { x, y }: { x: number; y: number },
    stone: Stone,
  ) => {
    // Calculate captured stone
    const capturedStones = getCapturedStones({
      x,
      y,
      stone,
      boardData: boardData.value,
    });

    // Update board
    updateBoard({ x, y, boardData: boardData.value, stone }, capturedStones);

    playStoneSound();

    histories.value = histories.value.concat({
      coordinate: { x, y },
      stone,
      capturedStones: capturedStones,
    });
  };

  const deleteLastHistory = () => {
    const lastHistory = histories.value.at(-1);
    if (!lastHistory) return;

    // Recover captured stones
    if (lastHistory.capturedStones) {
      lastHistory.capturedStones.forEach(({ x, y }) => {
        boardData.value[y][x].stone = lastHistory.stone === "X" ? "O" : "X";
      });
    }

    // Undo last move
    boardData.value[lastHistory.coordinate.y][lastHistory.coordinate.x].stone =
      ".";

    // Delete last history
    histories.value = histories.value.slice(0, -1);
    gameOver.value = false;
    playUndoSound();
    changeTurn(lastHistory.stone);
  };

  const addStoneToBoardData = (
    { x, y }: { x: number; y: number },
    stone: Stone,
  ) => {
    // Calculate captured stone
    const capturedStones = getCapturedStones({
      x,
      y,
      stone,
      boardData: boardData.value,
    });

    // Check double-three (double-three can be bypassed by capturing)
    if (
      capturedStones.length == 0 &&
      checkDoubleThree({ x, y, stone, boardData: boardData.value })
    ) {
      doAlert("Caution", "Double-three is not allowed", "Warn");
      return;
    }

    // Update board
    updateBoard({ x, y, boardData: boardData.value, stone }, capturedStones);

    // Add to history
    histories.value = histories.value.concat({
      coordinate: { x, y },
      stone,
      capturedStones: capturedStones,
    });

    // Check end condition
    const situation = {
      x,
      y,
      boardData: boardData.value,
      turn: stone,
      captured: {
        player1: player1TotalCaptured.value,
        player2: player2TotalCaptured.value,
        goal: settings.value.totalPairCaptured,
      },
    };
    const perfectFiveEnded = isPerfectFiveEnded(situation);

    if (perfectFiveEnded.result === GAME_END_SCENARIO.FIVE_OR_MORE_STONES) {
      gameOver.value = true;
      return doAlert(
        "Game Over",
        `${perfectFiveEnded.winner === "X" ? "Black" : "White"} Win - Five or more stones`,
        "Info",
      );
    }

    playStoneSound();
    changeTurn();

    nextTick(() => {
      // Check end condition after change turn
      // Check capture points
      situation.turn = turn.value;
      const captureEnded = isCaptureEnded(situation);
      if (captureEnded.result === GAME_END_SCENARIO.PAIR_CAPTURED) {
        gameOver.value = true;
        return doAlert(
          "Game Over",
          `${captureEnded.winner === "X" ? "Black" : "White"} Win - Captured ${situation.captured.goal}`,
          "Info",
        );
      }

      // Check current turn's five stones or more
      const currentTurnFiveEnded = isCurrentTurnFiveEnded(situation);
      if (
        currentTurnFiveEnded.result === GAME_END_SCENARIO.FIVE_OR_MORE_STONES
      ) {
        gameOver.value = true;
        return doAlert(
          "Game Over",
          `${currentTurnFiveEnded.winner === "X" ? "Black" : "White"} Win - Five or more stones`,
          "Info",
        );
      }

      // Check Draw
      const drawEnded = isDrawEnded(situation);
      if (drawEnded.result === GAME_END_SCENARIO.DRAW) {
        gameOver.value = true;
        return doAlert("Game Over", "Draw", "Info");
      }
    });
  };

  return {
    settings,
    turn,
    histories,
    boardData,
    gameOver,
    initGame,
    changeTurn,
    historyToLog,
    debugAddStoneToBoardData,
    addStoneToBoardData,
    deleteLastHistory,
    player1TotalCaptured,
    player2TotalCaptured,
  };
});
