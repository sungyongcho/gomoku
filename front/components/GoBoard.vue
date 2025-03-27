<script setup lang="ts">
const { boardData } = storeToRefs(useGameStore());

const onPutStone = ({ x, y }: { x: number; y: number }) => {
  emit("put", { x, y });
};
const onEvaluateStone = ({ x, y }: { x: number; y: number }) => {
  emit("evaluate", { x, y });
};
const emit = defineEmits(["put", "evaluate"]);
</script>
<template>
  <section
    class="mt-2 rounded-md bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-yellow-100 to-yellow-600 pl-5 pt-5"
  >
    <div v-for="(row, rowIndex) in boardData" :key="rowIndex" class="flex">
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
