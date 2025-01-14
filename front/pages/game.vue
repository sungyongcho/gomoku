<script setup lang="ts">
definePageMeta({
  layout: "game",
});

const { histories, turn } = storeToRefs(useGameStore());
const { deleteLastHistory, initGame, addStoneToBoardData } = useGameStore();

const onPutStone = ({ x, y }: { x: number; y: number }) => {
  addStoneToBoardData({ x, y }, turn.value);
};
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
        </div>
      </div>
      <InfoBoard />
    </div>
  </main>
</template>
