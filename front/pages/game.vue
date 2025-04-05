<script setup lang="ts">
import { useWebSocket } from "@vueuse/core";

import type {
  RequestType,
  SocketMoveResponse,
  SocketMoveRequest,
} from "~/types/game";

definePageMeta({
  layout: "game",
});

const {
  histories,
  turn,
  boardData,
  settings,
  evalScores,
  player1TotalCaptured,
  player2TotalCaptured,
  isAiThinking,
  gameOver,
} = storeToRefs(useGameStore());
const { deleteLastHistory, initGame, addStoneToBoardData } = useGameStore();

const lastHistory = computed(() => histories.value.at(-1));
const { doAlert } = useAlertStore();

const { data, send, close } = useWebSocket("ws://localhost:8005/ws/debug", {
  autoReconnect: {
    retries: 3,
    delay: 500,
    onFailed() {
      doAlert(
        "Error",
        "WebSocket connection failed. Please refresh the page to retry",
        "Warn",
      );
      isAiThinking.value = false;
    },
  },
});

const onPutStone = async ({ x, y }: { x: number; y: number }) => {
  const isSuccessToPutStone = await addStoneToBoardData({ x, y }, turn.value);
  await nextTick();

  if (isSuccessToPutStone && settings.value.isPlayer2AI && !gameOver.value) {
    onSendStone();
  }
};

const onSendData = (
  type: RequestType,
  coordinate?: { x: number; y: number },
) => {
  isAiThinking.value = true;
  send(
    JSON.stringify({
      type,
      difficulty: settings.value.difficulty,
      nextPlayer: lastHistory.value?.stone === "X" ? "O" : "X",
      goal: settings.value.totalPairCaptured,
      lastPlay: coordinate
        ? {
            coordinate: {
              x: coordinate.x,
              y: coordinate.y,
            },
            stone: lastHistory.value?.stone,
          }
        : undefined,
      board: boardData.value.map((row) => row.map((col) => col.stone)),
      scores: [
        { player: "X", score: player1TotalCaptured.value },
        { player: "O", score: player2TotalCaptured.value },
      ],
    } as SocketMoveRequest),
  );
};

const onSendStone = () => {
  onSendData(
    "move",
    lastHistory.value?.coordinate ? lastHistory.value.coordinate : undefined,
  );
};
const onEvaluateStone = (coordinate: undefined | { x: number; y: number }) => {
  if (coordinate) {
    onSendData("evaluate", coordinate);
  } else {
    // hide eval
    evalScores.value = [];
    data.value = null;
  }
};

const onRestart = () => {
  initGame();
  send(JSON.stringify({ type: "reset" }));
};

const purgeState = () => {
  isAiThinking.value = false;
  data.value = null;
};

watch(data, (rawData) => {
  if (!data.value) return;

  try {
    const res: SocketMoveResponse =
      typeof rawData === "string" ? JSON.parse(rawData) : rawData;

    if (res.type === "evaluate") {
      evalScores.value = res.evalScores;
      purgeState();
      return;
    }

    if (res.type === "error") {
      console.error(res);
      doAlert("Caution", res.error, "Warn");
      purgeState();
      return;
    }

    addStoneToBoardData(
      res.lastPlay.coordinate,
      res.lastPlay.stone,
      res.executionTime,
    );
  } catch (error) {
    console.error("Error processing WebSocket data:", error);
    doAlert(
      "Error",
      "An unexpected error occurred while processing data.",
      "Warn",
    );
  } finally {
    purgeState();
  }
});

onUnmounted(() => {
  close();
});
</script>
<template>
  <main
    class="relative h-[calc(100vh-80px)] w-full items-start justify-center -lg:h-[calc(100vh-68px)] lg:items-center"
  >
    <!-- Eval Stone -->
    <EvalTooltip />

    <!-- Board & History -->
    <div
      class="mx-auto flex h-[calc(100vh-48px)] max-w-[1280px] items-center justify-center gap-10 -lg:h-auto -lg:flex-col-reverse"
    >
      <div>
        <GoBoard @put="onPutStone" @evaluate="onEvaluateStone" />

        <div class="mt-3 flex w-full flex-wrap justify-center gap-3">
          <Button
            label="Undo a move"
            icon="pi pi-undo"
            :disabled="histories.length < 1"
            @click="deleteLastHistory"
          />
          <Button label="Restart" icon="pi pi-play" @click="onRestart" />
        </div>
      </div>
      <InfoBoard class="-lg:hidden" />
    </div>
  </main>
</template>
