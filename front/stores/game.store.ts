import { map, pipe, range, toArray } from "@fxts/core";
import { useStorage } from "@vueuse/core";
import { cloneDeep } from "lodash";

import {
  GAME_END_SCENARIO,
  type BoardInput,
  type BoardStone,
  type History,
  type Settings,
  type Stone,
  type StoneEval,
} from "~/types/game";

type GameSituation = {
  boardData: { stone: Stone }[][];
  x: number;
  y: number;
  turn: Stone;
  captured: {
    player1: number;
    player2: number;
    goal: number;
  };
};

export const useGameStore = defineStore("game", () => {
  const { doAlert } = useAlertStore();
  const { playStoneSound, playUndoSound } = useSound();
  const {
    getOppositeStone,
    getCapturedStones,
    checkDoubleThree,
    isCaptureEnded,
    isDrawEnded,
    isPerfectFiveEnded,
  } = useGameLogic();
  const settings = useStorage<Settings>("settings", {
    capture: true,
    doubleThree: true,
    totalPairCaptured: 5,
    firstMove: true,
    advantage1: 0,
    advantage2: 0,
    isPlayer2AI: true,
    isDebugTurnLocked: true,
    difficulty: "hard", // easy, medium, hard
    ai: "minmax", // minmax, alphago
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
  const currentHistoryIndex = ref(0);
  const onNextHistory = () => {
    const nextHistory = histories.value.at(currentHistoryIndex.value + 1);
    if (!nextHistory) return;

    // Apply captured stones
    if (nextHistory.capturedStones) {
      nextHistory.capturedStones.forEach(({ x, y, stone }) => {
        readOnlyBoardData.value[y][x].stone = ".";
      });
    }
    readOnlyBoardData.value[nextHistory.coordinate.y][
      nextHistory.coordinate.x
    ].stone = nextHistory.stone;

    currentHistoryIndex.value++;
    playStoneSound();
    changeTurn(nextHistory.stone);
  };
  const onPrevHistory = () => {
    const currentHistory = histories.value.at(currentHistoryIndex.value);
    if (!currentHistory) return;

    // Recover captured stones
    if (currentHistory.capturedStones) {
      currentHistory.capturedStones.forEach(({ x, y, stone }) => {
        readOnlyBoardData.value[y][x].stone = stone;
      });
    }

    // Undo last move
    readOnlyBoardData.value[currentHistory.coordinate.y][
      currentHistory.coordinate.x
    ].stone = ".";

    // Delete last history
    currentHistoryIndex.value--;
    gameOver.value = false;
    playUndoSound();
    changeTurn(currentHistory.stone);
  };

  const readOnlyHistories = computed(() =>
    histories.value.slice(0, currentHistoryIndex.value + 1),
  );

  // Auto selected history
  const _histories = computed(() => {
    return historyMode.value ? readOnlyHistories.value : histories.value;
  });

  const evalScores = ref<[StoneEval, StoneEval] | []>([]);
  const gameOver = useStorage<boolean>("gameOver", false);
  const historyMode = useStorage<boolean>("historyMode", false);

  const isAiThinking = ref(false);
  const boardData = useStorage<{ stone: Stone }[][]>(
    "boardData",
    initialBoard(),
  );

  const getPlayerTotalCaptured = (histories: History[], stone: Stone) => {
    return (
      histories
        .filter((h: History) => h.stone === stone)
        .reduce((acc: number, h: History) => {
          if (!h.capturedStones?.length) return acc;
          return h.capturedStones?.length / 2 + acc;
        }, 0) + settings.value.advantage1
    );
  };

  const readOnlyBoardData = useStorage<{ stone: Stone }[][]>(
    "readOnlyBoardData",
    initialBoard(),
  );

  watch(
    historyMode,
    (isHistoryMode) => {
      if (isHistoryMode) {
        currentHistoryIndex.value = histories.value.length - 1;
        readOnlyBoardData.value = cloneDeep(toRaw(boardData.value));
      }
    },
    { immediate: true },
  );

  const player1TotalCaptured = computed(() => {
    return getPlayerTotalCaptured(_histories.value, "X");
  });

  const player2TotalCaptured = computed(() => {
    return getPlayerTotalCaptured(_histories.value, "O");
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
    const history = `[${player}]: (${x}, ${y})`;
    if (h.executionTime) {
      return `${history} - ${h.executionTime.ms.toFixed(0)}ms`;
    }
    return history;
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

  const deleteLastHistory = () => {
    const lastHistory = histories.value.at(-1);
    if (!lastHistory) return;

    // Recover captured stones
    if (lastHistory.capturedStones) {
      lastHistory.capturedStones.forEach(({ x, y, stone }) => {
        boardData.value[y][x].stone = stone;
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

  // Check end condition after change turn
  const checkGameOver = (situation: GameSituation) => {
    // Check perfect five win
    for (let y = 0; y < boardData.value.length; y++) {
      for (let x = 0; x < boardData.value[y].length; x++) {
        const stone = boardData.value[y][x].stone;
        if (stone === ".") continue;

        const perfectFiveEnded = isPerfectFiveEnded({
          ...situation,
          x,
          y,
          turn: stone,
        });

        if (perfectFiveEnded.result === GAME_END_SCENARIO.FIVE_OR_MORE_STONES) {
          gameOver.value = true;
          return doAlert(
            "Game Over",
            `${perfectFiveEnded.winner === "X" ? "Black" : "White"} Win - Five or more stones`,
            "Info",
          );
        }
      }
    }

    // Check capture points
    const captureEnded = isCaptureEnded(situation);
    if (captureEnded.result === GAME_END_SCENARIO.PAIR_CAPTURED) {
      gameOver.value = true;
      return doAlert(
        "Game Over",
        `${captureEnded.winner === "X" ? "Black" : "White"} Win - Captured ${situation.captured.goal}`,
        "Info",
      );
    }

    // Check Draw
    const drawEnded = isDrawEnded(situation);
    if (drawEnded.result === GAME_END_SCENARIO.DRAW) {
      gameOver.value = true;
      return doAlert("Game Over", "Draw", "Info");
    }
  };

  const debugAddStoneToBoardData = async (
    { x, y }: { x: number; y: number },
    stone: Stone,
    askAi: boolean,
    executionTime?: { s: number; ms: number; ns: number },
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
      executionTime,
    });

    // Check end condition
    const situation: GameSituation = {
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

    playStoneSound();

    const oppositeStone = getOppositeStone(stone);
    if (askAi || !settings.value.isDebugTurnLocked) {
      changeTurn();
    }

    await nextTick();
    checkGameOver({ ...situation, turn: oppositeStone });
  };

  // Add stone to board
  // Return true or false to determine whether we send data to AI
  // If return true, means we success to put stone, so we need to send
  // If return false, means we fail to put stone, so we don't need to send
  const addStoneToBoardData = async (
    { x, y }: { x: number; y: number },
    stone: Stone,
    executionTime?: { s: number; ms: number; ns: number },
  ): Promise<boolean> => {
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
      return false;
    }

    // Update board
    updateBoard({ x, y, boardData: boardData.value, stone }, capturedStones);

    // Add to history
    histories.value = histories.value.concat({
      coordinate: { x, y },
      stone,
      capturedStones: capturedStones,
      executionTime,
    });

    // Check end condition
    const situation: GameSituation = {
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

    playStoneSound();
    changeTurn();

    await nextTick();
    checkGameOver({ ...situation, turn: turn.value });
    return true;
  };

  const exportUrl = (): string => {
    // boardData, histories, settings
    const route = useRoute();
    const data = {
      boardData: boardData.value,
      histories: histories.value,
      settings: settings.value,
      turn: turn.value,
      gameOver: gameOver.value,
    };
    const dataStr = JSON.stringify(data);
    const base64Encoded = btoa(dataStr);
    return `${window.location.origin}${route.fullPath}?data=${base64Encoded}`;
  };

  const exportJson = () => {
    const data = {
      boardData: boardData.value,
      histories: histories.value,
      settings: settings.value,
      turn: turn.value,
      gameOver: gameOver.value,
    };
    pipe(
      data,
      (d) =>
        "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(d)),
      (fileString) => {
        const a = document.createElement("a");
        a.setAttribute("href", fileString);
        a.setAttribute("download", "game.json");
        document.body.appendChild(a); // required for firefox
        a.click();
        a.remove();
      },
    );
  };

  const importGame = (data: {
    boardData: { stone: Stone }[][];
    histories: History[];
    settings: Settings;
    turn: Stone;
    gameOver: boolean;
  }) => {
    boardData.value = data.boardData;
    histories.value = data.histories;
    settings.value = data.settings;
    turn.value = data.turn;
    gameOver.value = data.gameOver;
  };
  const importData = (dataStr: string) => {
    const data = JSON.parse(atob(dataStr));
    importGame(data);
  };

  return {
    settings,
    turn,
    _histories,
    histories, // game mode history
    readOnlyHistories, // history mode history
    historyMode,
    onPrevHistory,
    onNextHistory,
    boardData,
    readOnlyBoardData,
    gameOver,
    initGame,
    changeTurn,
    historyToLog,
    debugAddStoneToBoardData,
    addStoneToBoardData,
    deleteLastHistory,
    player1TotalCaptured,
    player2TotalCaptured,
    evalScores,
    isAiThinking,
    exportUrl,
    importGame,
    importData,
    exportJson,
    initialBoard,
    getPlayerTotalCaptured,
  };
});
