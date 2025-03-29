<script setup lang="ts">
import player1Image from "~/assets/player1.webp";
import player2Image from "~/assets/player2.webp";
import aiImage from "~/assets/ai.webp";

const {
  histories,
  player1TotalCaptured,
  player2TotalCaptured,
  settings,
  turn,
} = storeToRefs(useGameStore());
const { historyToLog } = useGameStore();
const historyEl = ref<HTMLElement>();
const route = useRoute();
const isDebug = computed(() => route.name === "debug");
const aiHistories = computed(() => {
  return histories.value.filter((h) => !!h.executionTime?.ms);
});

const aiAverageResponseTime = computed(() => {
  return (
    aiHistories.value.reduce((a, b) => a + (b.executionTime?.ms ?? 0), 0) /
    (aiHistories.value.length || 1)
  ).toFixed(0);
});

watch(
  () => histories.value?.length,
  () => {
    console.log("HERE");
    nextTick(() => {
      const scrollHeight = historyEl.value!.scrollHeight;
      historyEl.value!.scroll(0, scrollHeight);
    });
  },
  { immediate: true },
);
</script>
<template>
  <aside class="flex w-[300px] flex-col items-center gap-5">
    <section class="flex w-full items-center justify-between gap-10 -lg:hidden">
      <button
        class="flex flex-col items-center justify-center border-4 border-transparent p-2"
        :class="{
          ['border-yellow-500']: turn === 'X',
        }"
        :disabled="!isDebug"
        @click="turn = 'X'"
      >
        <Avatar
          size="xlarge"
          shape="circle"
          :image="player1Image"
          class="border-4 border-black shadow-[0_0_4px_1px_gray]"
        />
        <span>Player1</span>
      </button>

      <span class="text-2xl">vs</span>

      <button
        class="flex flex-col items-center justify-center border-4 border-transparent p-2"
        :class="{
          ['border-yellow-500']: turn === 'O',
        }"
        :disabled="!isDebug"
        @click="turn = 'O'"
      >
        <Avatar
          class="border-4 border-gray-100 shadow-[0_0_4px_1px_black]"
          size="xlarge"
          shape="circle"
          :image="settings.isPlayer2AI ? aiImage : player2Image"
        />
        <span> {{ settings.isPlayer2AI ? "AI" : "Player2" }}</span>
      </button>
    </section>

    <CapturedScore
      class="w-full -lg:hidden"
      :player1-total-captured="player1TotalCaptured"
      :player2-total-captured="player2TotalCaptured"
    />

    <section class="w-full -lg:hidden">
      <p class="mb-1 pl-1 text-sm">Game history</p>
      <div
        ref="historyEl"
        class="relative h-[50vh] w-full overflow-y-auto rounded-md bg-black p-2 pt-8 text-sm"
      >
        <p
          class="absolute left-0 top-0 w-full rounded-t border-[4px] border-black bg-gray-300 px-2 text-black"
        >
          AI Average Response Time:
          {{ aiAverageResponseTime }}
          ms
        </p>
        <p
          v-for="(h, index) in histories"
          :key="index"
          class="text-gray-500 last:text-white"
        >
          {{ historyToLog(h) }}
        </p>
      </div>
    </section>
  </aside>
</template>
