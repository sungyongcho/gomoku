<script setup lang="ts">
import { useWebSocket } from "@vueuse/core";
import type { SocketMoveResponse } from "~/types/game";

definePageMeta({
  layout: "game",
});

const { histories, turn, boardData } = storeToRefs(useGameStore());
const { deleteLastHistory, initGame, debugAddStoneToBoardData } =
  useGameStore();
const onPutStone = ({ x, y }: { x: number; y: number }) => {
  debugAddStoneToBoardData({ x, y }, turn.value);
};
const lastHistory = computed(() => histories.value.at(-1));
const { status, data, send, open, close } = useWebSocket(
  "ws://localhost:8000/ws/gomoku",
);

const onSendData = () => {
  send(
    JSON.stringify({
      type: "move",
      player: turn.value === "X" ? "O" : "X",
      board: boardData.value.map((row) =>
        row.map((col) => (col.stone === "" ? "." : col.stone)),
      ),
    }),
  );
};

watch(data, (res: SocketMoveResponse) => {
  if (res.status === "error") {
    // Handle Error
    console.error("Error from socket server");
    return;
  }

  boardData.value = res.board.map((row) => row.map((col) => ({ stone: col })));
  turn.value = res.player === "X" ? "O" : "X";
  histories.value = histories.value.concat({
    coordinate: { x: res.newStone.x, y: res.newStone.y },
    stone: res.player,
    capturedStones: res.capturedStones,
  });
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
