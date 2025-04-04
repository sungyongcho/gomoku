<script setup lang="ts">
import { useWebSocket } from "@vueuse/core";
import { useMouse } from "@vueuse/core";

import type {
  RequestType,
  SocketMoveResponse,
  SocketMoveRequest,
} from "~/types/game";

definePageMeta({
  layout: "debug",
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
} = storeToRefs(useGameStore());
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

const { x, y } = useMouse({ touch: false });
const lastHistory = computed(() => histories.value.at(-1));
const { deleteLastHistory, initGame, debugAddStoneToBoardData } =
  useGameStore();
const { doAlert } = useAlertStore();
const onPutStone = ({ x, y }: { x: number; y: number }) => {
  debugAddStoneToBoardData({ x, y }, turn.value);
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

    debugAddStoneToBoardData(
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
    class="flex h-[calc(100vh-80px)] w-full items-start justify-center -lg:h-[calc(100vh-68px)] lg:items-center"
  >
    <!-- Eval Stone -->
    <div
      v-if="evalScores.length"
      class="shadow-3xl fixed left-0 top-0 z-[100] flex flex-col gap-2 rounded-md bg-white p-2 shadow-xl"
      :style="{ transform: `translate(${x + 20}px, ${y - 30}px)` }"
    >
      <div v-for="s in evalScores" class="flex items-center gap-2">
        <i
          class="pi"
          :class="{
            ['pi-circle-fill']: s.player === 'X',
            ['pi-circle']: s.player === 'O',
          }"
        ></i>
        <EvalScoreGage v-model="s.rating" />
      </div>
    </div>

    <!-- Board & History -->
    <div
      class="flex max-w-[1280px] items-center justify-center gap-10 -lg:flex-col-reverse"
    >
      <div>
        <GoBoard @put="onPutStone" @evaluate="onEvaluateStone" />

        <div class="mt-3 flex w-full justify-center gap-3">
          <Button
            label="Undo a move"
            icon="pi pi-undo"
            :disabled="histories.length < 1"
            @click="deleteLastHistory"
          />
          <Button label="Restart" icon="pi pi-play" @click="onRestart" />
          <Button
            label="Send"
            icon="pi pi-send"
            @click="onSendStone"
            :disabled="isAiThinking"
            :loading="isAiThinking"
          />
          <ToggleButton
            onIcon="pi pi-lock"
            offIcon="pi pi-lock-open"
            v-model="settings.isDebugTurnLocked"
            onLabel="Turn Locked"
            offLabel="Turn Unlocked"
          />
        </div>
      </div>
      <InfoBoard />
    </div>
  </main>
</template>
