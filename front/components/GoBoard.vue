<script setup lang="ts">
const { boardData, historyMode, readOnlyBoardData } =
  storeToRefs(useGameStore());

const _boardData = computed(() =>
  historyMode.value ? readOnlyBoardData.value : boardData.value,
);
const onPutStone = ({ x, y }: { x: number; y: number }) => {
  emit("put", { x, y });
};
const onEvaluateStone = (coordinate: { x: number; y: number } | undefined) => {
  emit("evaluate", coordinate);
};
const emit = defineEmits(["put", "evaluate"]);
</script>
<template>
  <section
    class="mt-2 rounded-md bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-yellow-100 to-yellow-600 pl-5 pt-5 -lg:mt-4 -sm:pl-3 -sm:pt-3"
    :class="{
      'pointer-events-none bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] !from-stone-100 !to-stone-600':
        historyMode,
    }"
  >
    <div v-for="(row, rowIndex) in _boardData" :key="rowIndex" class="flex">
      <div v-for="(cell, cellIndex) in row" :key="cellIndex" class="flex">
        <Cell
          :x="cellIndex"
          :y="rowIndex"
          @put="onPutStone"
          @evaluate="onEvaluateStone"
          :stone="cell.stone"
        />
      </div>
    </div>
  </section>
</template>
