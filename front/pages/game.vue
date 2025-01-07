<script setup lang="ts">
definePageMeta({
  layout: "game",
});

const { histories, boardData } = storeToRefs(useGameStore());
const { showGameOverIfWinnerExists, deleteLastHistory, initGame } =
  useGameStore();

watch(
  () => histories.value,
  (newHistory, oldHistory) => {
    if (newHistory.length > 0 && newHistory.length > oldHistory.length) {
      const prevHistory = oldHistory.at(-1);
      if (!prevHistory) return;

      nextTick(() => {
        showGameOverIfWinnerExists(
          {
            x: prevHistory.coordinate.x,
            y: prevHistory.coordinate.y,
            stone: prevHistory.stoneType,
            boardData: boardData.value,
          },
          false,
        );
      });
    }
  },
);
</script>
<template>
  <main
    class="flex h-[calc(100vh-80px)] w-full items-start justify-center -lg:h-[calc(100vh-68px)] lg:items-center"
  >
    <div
      class="flex max-w-[1280px] items-center justify-center gap-10 -lg:flex-col-reverse"
    >
      <div>
        <GoBoard />
        <div class="mt-3 flex w-full justify-center gap-3">
          <Button
            label="Undo a move"
            icon="pi pi-undo"
            :disabled="histories.length < 1"
            @click="deleteLastHistory"
          />
          <Button label="Restart" icon="pi pi-play" @click="initGame" />
        </div>
      </div>
      <InfoBoard />
    </div>
  </main>
</template>
