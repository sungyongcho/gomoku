<script setup lang="ts">
import { useWebSocket } from "@vueuse/core";

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
  _histories,
  turn,
  boardData,
  settings,
  evalScores,
  player1TotalCaptured,
  player2TotalCaptured,
  isAiThinking,
  historyMode,
} = storeToRefs(useGameStore());
const {
  deleteLastHistory,
  initGame,
  debugAddStoneToBoardData,
  onNextHistory,
  onPrevHistory,
} = useGameStore();

const lastHistory = computed(() => histories.value.at(-1));
const { doAlert, closeAlert } = useAlertStore();
const maintenance = useMaintenanceStore();
const { getDebugSocketUrl } = useEnv();

const showReconnectAlert = (message?: string) => {
  doAlert({
    header: "Error",
    message: message || "WebSocket connection failed. Reconnecting...",
    type: "Warn",
    actionIcon: "pi pi-undo",
    actionLabel: "Reconnect now",
    action: () => {
      open();
      closeAlert();
    },
  });
};

const { data, send, close, open, status } = useWebSocket(getDebugSocketUrl(), {
});

const { ensureOpen, startRequestTimeout, clearRequestTimeout, scheduleReconnect } =
  useWebSocketReliability({
    status,
    open,
    close,
    onRequestTimedOut(type) {
      isAiThinking.value = false;
      showReconnectAlert(
        `No response from backend for '${type}' request. Auto reconnecting...`,
      );
    },
  });

watch(
  status,
  (s) => {
    if (s === "OPEN") {
      closeAlert();
      return;
    }

    if (s === "CLOSED") {
      isAiThinking.value = false;
      scheduleReconnect();
    }
  },
  { immediate: true },
);

const onPutStone = ({ x, y }: { x: number; y: number }) => {
  debugAddStoneToBoardData({ x, y }, turn.value, false);
};

const onSendData = (
  type: RequestType,
  coordinate?: { x: number; y: number },
) => {
  if (!ensureOpen()) {
    isAiThinking.value = false;
    showReconnectAlert();
    return;
  }

  isAiThinking.value = true;
  startRequestTimeout(type);
  send(
    JSON.stringify({
      type,
      difficulty: settings.value.difficulty,
      nextPlayer: lastHistory.value?.stone === "X" ? "O" : "X",
      goal: settings.value.totalPairCaptured,
      enableCapture: settings.value.enableCapture,
      enableDoubleThreeRestriction: settings.value.enableDoubleThreeRestriction,
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
    clearRequestTimeout();
    isAiThinking.value = false;
    evalScores.value = [];
    data.value = null;
  }
};

const onRestart = () => {
  initGame();
  if (!ensureOpen()) {
    showReconnectAlert();
    return;
  }
  send(JSON.stringify({ type: "reset" }));
};

const purgeState = () => {
  clearRequestTimeout();
  isAiThinking.value = false;
  data.value = null;
};

watch(data, (rawData) => {
  if (!data.value) return;
  maintenance.reportBackendSuccess();

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
      doAlert({
        header: "Caution",
        message: res.error as string,
        type: "Warn",
      });
      purgeState();
      return;
    }

    debugAddStoneToBoardData(
      res.lastPlay.coordinate,
      res.lastPlay.stone,
      true, // isAi
      res.executionTime,
    );
  } catch (error) {
    console.error("Error processing WebSocket data:", error);
    doAlert({
      header: "Error",
      message: "An unexpected error occurred while processing data.",
      type: "Warn",
    });
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
    class="relative h-screen w-full items-start justify-center lg:items-center"
  >
    <!-- Eval Stone -->
    <EvalTooltip />

    <!-- Board & History -->
    <div
      class="mx-auto flex h-screen max-w-[1280px] items-center justify-center gap-10 -lg:flex-col-reverse"
    >
      <div>
        <GoBoard
          @put="onPutStone"
          @evaluate="onEvaluateStone"
          :boardData="boardData"
        />

        <div class="mt-3 flex w-full flex-wrap justify-center gap-3">
          <template v-if="!historyMode">
            <Button
              label="Undo a move"
              icon="pi pi-undo"
              size="small"
              :disabled="histories.length < 1"
              @click="deleteLastHistory"
            />
            <Button
              label="Restart"
              size="small"
              icon="pi pi-play"
              @click="onRestart"
            />
            <Button
              label="Send"
              icon="pi pi-send"
              @click="onSendStone"
              :disabled="isAiThinking"
              :loading="isAiThinking"
              size="small"
            />
            <ToggleButton
              onIcon="pi pi-lock"
              offIcon="pi pi-lock-open"
              v-model="settings.isDebugTurnLocked"
              onLabel="Turn Locked"
              offLabel="Turn Unlocked"
              size="small"
            />
          </template>
          <template v-else>
            <Button
              label="Prev"
              icon="pi pi-arrow-left"
              :disabled="_histories.length === 0"
              @click="onPrevHistory"
              size="small"
            />
            <Button
              label="Next"
              icon-pos="right"
              icon="pi pi-arrow-right"
              @click="onNextHistory"
              :disabled="_histories.length === histories.length"
              size="small"
            />
          </template>
        </div>
      </div>
      <InfoBoard class="-lg:hidden" />
    </div>
  </main>
</template>
