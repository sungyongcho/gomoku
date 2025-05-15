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
const { getSocketUrl } = useEnv();

const { data, send, close, open, status } = useWebSocket(getSocketUrl(), {
  autoReconnect: {
    retries: 0,
    onFailed() {
      doAlert({
        header: "Error",
        message: "WebSocket connection failed. Click button to reconnect",
        type: "Warn",
        actionIcon: "pi pi-undo",
        actionLabel: "Reconnect",
        action: () => {
          open();
          closeAlert();
        },
      });
      isAiThinking.value = false;
    },
  },
});

const onPutStone = ({ x, y }: { x: number; y: number }) => {
  debugAddStoneToBoardData({ x, y }, turn.value, false);
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
  if (status.value === "CLOSED") {
    doAlert({
      header: "Error",
      message: "WebSocket connection failed. Click button to reconnect",
      type: "Warn",
      actionIcon: "pi pi-undo",
      actionLabel: "Reconnect",
      action: () => {
        open();
        nextTick(() => {
          closeAlert();
        });
      },
    });
    return;
  }

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
    class="relative h-[calc(100vh-80px)] w-full items-start justify-center -lg:h-[calc(100vh-68px)] lg:items-center"
  >
    <!-- Eval Stone -->
    <EvalTooltip />

    <!-- Board & History -->
    <div
      class="mx-auto flex h-[calc(100vh-48px)] max-w-[1280px] items-center justify-center gap-10 -lg:h-auto -lg:flex-col-reverse"
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
          </template>
          <template v-else>
            <Button
              label="Prev"
              icon="pi pi-arrow-left"
              :disabled="_histories.length === 0"
              @click="onPrevHistory"
            />
            <Button
              label="Next"
              icon-pos="right"
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
