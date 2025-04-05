<script setup lang="ts">
import { useMouse } from "@vueuse/core";

const { x, y } = useMouse({ touch: false });
const { evalScores } = storeToRefs(useGameStore());
</script>
<template>
  <div
    v-if="evalScores.length"
    class="shadow-3xl fixed left-0 top-0 z-[100] flex flex-col gap-2 rounded-md bg-white p-2 shadow-xl"
    :style="{ transform: `translate(${x + 20}px, ${y - 30}px)` }"
  >
    <div v-for="s in evalScores" class="flex items-center gap-2">
      <i
        class="pi"
        :class="{
          ['pi-circle-fill']: s.player === 'X',
          ['pi-circle']: s.player === 'O',
        }"
      ></i>
      <EvalScoreGage v-model="s.rating" />
    </div>
  </div>
</template>
