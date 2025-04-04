<script setup lang="ts">
import player1Image from "~/assets/player1.webp";
import player2Image from "~/assets/player2.webp";
import aiImage from "~/assets/ai.webp";
import { useClipboard } from "@vueuse/core";

const {
  histories,
  player1TotalCaptured,
  player2TotalCaptured,
  settings,
  turn,
  isAiThinking,
} = storeToRefs(useGameStore());
const { historyToLog, exportData } = useGameStore();
const historyEl = ref<HTMLElement>();
const route = useRoute();
const isDebug = computed(() => route.name === "debug");
const exportedData = ref("");
const { text, copy, copied, isSupported } = useClipboard({
  source: exportedData,
});

const onClickExport = () => {
  const url = exportData();
  copy(url);
};
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
        <InfoAvatar
          :image="player1Image"
          :loading="isAiThinking && turn === 'X'"
          size="xlarge"
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
        <InfoAvatar
          :image="settings.isPlayer2AI ? aiImage : player2Image"
          :loading="isAiThinking && turn === 'O'"
          size="xlarge"
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
      <div class="mb-1 flex items-center justify-between pl-1 text-sm">
        <p class="text-md font-bold">Game history</p>

        <ClientOnly>
          <button
            v-if="isSupported"
            class="rounded-md bg-black px-2 py-1 text-white hover:bg-opacity-80"
            @click="onClickExport"
          >
            <span v-if="!copied" class="flex items-center gap-1">
              Export as url <i class="pi pi-link"></i>
            </span>
            <span v-else>Copied!</span>
          </button>
        </ClientOnly>
      </div>
      <div
        ref="historyEl"
        class="relative h-[50vh] w-full overflow-y-auto rounded-md bg-black text-sm"
      >
        <p
          class="sticky left-0 top-0 w-full rounded-t border-[4px] border-black bg-gray-300 px-2 text-black"
        >
          AI Average Response Time:
          {{ aiAverageResponseTime }}
          ms
        </p>
        <ul class="pb-2">
          <li
            v-for="(h, index) in histories"
            :key="index"
            class="px-2 text-gray-500 last:text-white"
          >
            {{ historyToLog(h) }}
          </li>
        </ul>
      </div>
    </section>
  </aside>
</template>
