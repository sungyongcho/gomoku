<script setup lang="ts">
import { useWebSocket } from "@vueuse/core";

import type { SocketDebugMoveResponse, SocketMoveRequest } from "~/types/game";

definePageMeta({
  layout: "debug",
});

const {
  histories,
  turn,
  boardData,
  settings,
  player1TotalCaptured,
  player2TotalCaptured,
} = storeToRefs(useGameStore());
const { deleteLastHistory, initGame, debugAddStoneToBoardData } =
  useGameStore();
const { doAlert } = useAlertStore();
const onPutStone = ({ x, y }: { x: number; y: number }) => {
  debugAddStoneToBoardData({ x, y }, turn.value);
};
const lastHistory = computed(() => histories.value.at(-1));
const { status, data, send, open, close } = useWebSocket(
  "ws://localhost:8000/ws/debug",
);

const onSendData = () => {
  send(
    JSON.stringify({
      type: "move",
      nextPlayer: turn.value === "X" ? "O" : "X",
      goal: settings.value.totalPairCaptured,
      lastPlay: lastHistory.value
        ? {
            coordinate: {
              x: lastHistory.value?.coordinate.x,
              y: lastHistory.value?.coordinate.y,
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

watch(data, (rawData) => {
  try {
    const res: SocketDebugMoveResponse =
      typeof rawData === "string" ? JSON.parse(rawData) : rawData;
    if (res.type === "error") {
      if (res.status === "tba") {
        doAlert("Caution", "TBA", "Warn");
        return;
      }
      doAlert("Caution", "Double-three is not allowed", "Warn");
      return;
    }

    boardData.value = res.board.map((row) =>
      row.map((col) => ({ stone: col })),
    );
    turn.value = res.lastPlay.stone === "X" ? "O" : "X";
    histories.value = histories.value.concat({
      coordinate: res.lastPlay.coordinate,
      stone: res.lastPlay.stone,
      capturedStones: res.capturedStones,
    });
  } catch (error) {
    console.error("Error processing WebSocket data:", error);
    doAlert(
      "Error",
      "An unexpected error occurred while processing data.",
      "Warn",
    );
  }
});
</script>
<template>
  <main
    class="flex h-[calc(100vh-80px)] w-full items-start justify-center -lg:h-[calc(100vh-68px)] lg:items-center"
  >
    <div
      class="flex max-w-[1280px] items-center justify-center gap-10 -lg:flex-col-reverse"
    >
      <div>
        <GoBoard @put="onPutStone" />

        <div class="mt-3 flex w-full justify-center gap-3">
          <Button
            label="Undo a move"
            icon="pi pi-undo"
            :disabled="histories.length < 1"
            @click="deleteLastHistory"
          />
          <Button label="Restart" icon="pi pi-play" @click="initGame" />
          <Button label="Send" icon="pi pi-send" @click="onSendData" />
        </div>
      </div>
      <DebugInfoBoard />
    </div>
  </main>
</template>
