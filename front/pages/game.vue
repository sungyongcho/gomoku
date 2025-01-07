<script setup lang="ts">
definePageMeta({
  layout: "game",
});

const { histories, boardData } = storeToRefs(useGameStore());
const { showGameOverIfWinnerExists } = useGameStore();
watch(
  () => histories.value,
  (newHistory, oldHistory) => {
    if (newHistory.length > 0 && newHistory.length > oldHistory.length) {
      const prevHistory = oldHistory.at(-1);
      if (!prevHistory) return;

      console.log("After History added");
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
    class="flex h-[calc(100vh-68px)] w-full items-center justify-center -sm:h-[calc(100vh-128px)]"
  >
    <div
      class="flex max-w-[1280px] items-center justify-center gap-10 -lg:flex-col-reverse"
    >
      <GoBoard />
      <InfoBoard />
    </div>
  </main>
</template>
