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
  _histories,
  turn,
  boardData,
  settings,
  evalScores,
  player1TotalCaptured,
  player2TotalCaptured,
  isAiThinking,
  historyMode,
  gameOver,
} = storeToRefs(useGameStore());
const {
  deleteLastTurn,
  initGame,
  addStoneToBoardData,
  onPrevHistory,
  onNextHistory,
} = useGameStore();

const lastHistory = computed(() => histories.value.at(-1));

const shouldAiPlayFirstMove = computed(
  () =>
    histories.value.length === 0 &&
    settings.value.firstMove === "Player2" &&
    settings.value.isPlayer2AI,
);

const aiFirstMoveRequested = ref(false);

const requestAiFirstMoveIfNeeded = () => {
  if (
    !shouldAiPlayFirstMove.value ||
    status.value !== "OPEN" ||
    aiFirstMoveRequested.value
  )
    return;
  aiFirstMoveRequested.value = true;
  onSendStone();
};

const canUndoTurn = computed(() => {
  if (gameOver.value || histories.value.length < 1) return false;
  if (settings.value.isPlayer2AI) {
    return !isAiThinking.value && histories.value.at(-1)?.stone === "O";
  }
  return true;
});
const { doAlert, closeAlert } = useAlertStore();
const maintenance = useMaintenanceStore();
const { getSocketUrl } = useEnv();

const socketUrl = computed(() => getSocketUrl());

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

const { data, send, close, status, open } = useWebSocket(socketUrl, {
  onConnected() {
    requestAiFirstMoveIfNeeded();
  },
});

const {
  ensureOpen,
  startRequestTimeout,
  clearRequestTimeout,
  scheduleReconnect,
} = useWebSocketReliability({
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
  () => histories.value.length,
  (len) => {
    if (len === 0) aiFirstMoveRequested.value = false;
  },
);

watch(
  status,
  (s) => {
    if (s === "OPEN") {
      closeAlert();
      requestAiFirstMoveIfNeeded();
      return;
    }

    if (s === "CLOSED") {
      isAiThinking.value = false;
      scheduleReconnect();
    }
  },
  { immediate: true },
);

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
  if (settings.value.isPlayer2AI && settings.value.firstMove === "Player2") {
    onSendStone();
  }
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

    addStoneToBoardData(
      res.lastPlay.coordinate,
      res.lastPlay.stone,
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
              :label="settings.isPlayer2AI ? 'Undo a turn' : 'Undo a move'"
              size="small"
              icon="pi pi-undo"
              :disabled="!canUndoTurn"
              @click="deleteLastTurn"
            />
            <Button
              size="small"
              label="Restart"
              icon="pi pi-play"
              @click="onRestart"
            />
          </template>
          <template v-else>
            <Button
              label="Prev"
              size="small"
              icon="pi pi-arrow-left"
              :disabled="_histories.length === 0"
              @click="onPrevHistory"
            />
            <Button
              label="Next"
              icon-pos="right"
              size="small"
              icon="pi pi-arrow-right"
              @click="onNextHistory"
              :disabled="_histories.length === histories.length"
            />
          </template>
        </div>
      </div>
      <InfoBoard class="-lg:hidden" />
    </div>
  </main>
</template>
