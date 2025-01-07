<script setup lang="ts">
const { turn, boardData } = storeToRefs(useGameStore());
const { addStoneToBoardData } = useGameStore();

const onPutStone = ({ x, y }: { x: number; y: number }) => {
  addStoneToBoardData({ x, y }, turn.value);
};
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
          :stoneType="cell.stoneType"
        />
      </div>
    </div>
  </section>
</template>
