<script setup lang="ts">
const props = defineProps({
  player1TotalCaptured: {
    type: Number,
    default: 0,
  },
  player2TotalCaptured: {
    type: Number,
    default: 0,
  },
});

const { settings } = storeToRefs(useGameStore());
</script>
<template>
  <section
    class="flex items-center justify-between gap-10 rounded-md border-2 p-2"
  >
    <div class="flex flex-col items-center justify-center">
      <div
        class="flex h-[52px] w-[52px] items-end rounded-lg bg-black p-2 text-white"
      >
        <span class="mr-1 text-3xl -sm:text-xl">{{
          player1TotalCaptured
        }}</span>
        <small class="text-sm text-gray-300">
          /{{ settings.totalPairCaptured }}
        </small>
      </div>
    </div>

    <span class="text-body text-center">Pair captured</span>

    <div class="flex flex-col items-center justify-center">
      <div
        class="flex h-[52px] w-[52px] items-end rounded-lg bg-black p-2 text-white"
      >
        <span class="mr-1 text-3xl -sm:text-xl">{{
          player2TotalCaptured
        }}</span>
        <small class="text-sm text-gray-300">
          /{{ settings.totalPairCaptured }}
        </small>
      </div>
    </div>
  </section>
</template>
