<script setup lang="ts">
import player1Image from "~/assets/player1.webp";
import player2Image from "~/assets/player2.webp";
import aiImage from "~/assets/ai.webp";
import { useClipboard } from "@vueuse/core";

const {
  _histories,
  player1TotalCaptured,
  player2TotalCaptured,
  settings,
  turn,
  historyMode,
  isAiThinking,
} = storeToRefs(useGameStore());
const { historyToLog, exportUrl, exportJson } = useGameStore();
const historyEl = ref<HTMLElement>();
const route = useRoute();
const isDebug = computed(() => route.name === "debug");
const exportedData = ref("");
const { copy, copied, isSupported } = useClipboard({
  source: exportedData,
  legacy: true,
});

const onExportUrl = () => {
  const url = exportUrl();
  copy(url);
};
const aiHistories = computed(() => {
  return _histories.value.filter((h) => !!h.executionTime?.ms);
});

const onExportJson = () => {
  exportJson();
};

const aiAverageResponseTime = computed(() => {
  return (
    aiHistories.value.reduce((a, b) => a + (b.executionTime?.ms ?? 0), 0) /
    (aiHistories.value.length || 1)
  ).toFixed(0);
});

watch(
  () => [_histories.value?.length],
  () => {
    nextTick(() => {
      if (!historyEl.value) return;
      const scrollHeight = historyEl.value!.scrollHeight;
      historyEl.value!.scroll(0, scrollHeight);
    });
  },
  { immediate: true },
);
</script>
<template>
  <aside class="flex flex-col items-center gap-5 p-2 -sm:gap-3">
    <section class="flex w-full items-center justify-between gap-10">
      <button
        class="flex flex-col items-center justify-center"
        :disabled="!isDebug"
        @click="turn = 'X'"
      >
        <InfoAvatar
          :image="player1Image"
          :loading="isAiThinking && turn === 'X'"
          size="xlarge"
          color="black"
          :active="turn === 'X'"
        />
        <span>Player1</span>
      </button>

      <div class="flex flex-col items-center">
        <span class="text-2xl">vs</span>
        <span class="text-center text-sm">
          Turn {{ Math.floor(_histories.length / 2) }}
        </span>
      </div>

      <button
        class="flex flex-col items-center justify-center p-2"
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
          color="white"
          :active="turn === 'O'"
        />
        <span> {{ settings.isPlayer2AI ? "AI" : "Player2" }}</span>
      </button>
    </section>

    <CapturedScore
      class="w-full"
      :player1-total-captured="player1TotalCaptured"
      :player2-total-captured="player2TotalCaptured"
    />

    <section class="w-full">
      <div
        class="flex items-center justify-between rounded-t-md bg-gray-200 p-2 text-sm"
      >
        <button
          class="text-md rounded-md bg-gray-200 px-2 py-1 text-black hover:bg-gray-300"
          :class="{ ['bg-yellow-400 hover:bg-yellow-500']: historyMode }"
          @click="historyMode = !historyMode"
        >
          Game history
        </button>
        <ClientOnly>
          <div class="flex gap-1">
            <button
              v-if="isSupported"
              class="rounded-md bg-black px-2 py-1 text-white hover:bg-opacity-80"
              @click="onExportUrl"
            >
              <span v-if="!copied" class="flex items-center gap-1">
                url <i class="pi pi-link"></i>
              </span>
              <span v-else>Copied!</span>
            </button>
            <button
              @click="onExportJson"
              class="rounded-md bg-black px-2 py-1 text-white hover:bg-opacity-80"
            >
              json <i class="pi pi-download"></i>
            </button>
          </div>
        </ClientOnly>
      </div>
      <div
        ref="historyEl"
        class="h-[50vh] w-full overflow-y-auto bg-black text-sm"
      >
        <ul class="py-2">
          <li
            v-for="(h, index) in _histories"
            :key="index"
            class="px-2 text-gray-500 last:text-white"
          >
            {{ historyToLog(h) }}
          </li>
        </ul>
      </div>
      <p
        class="left-0 top-0 m-0 w-full rounded-b-md bg-gray-200 p-2 px-2 text-sm font-bold text-black"
      >
        AI Average Time:
        {{ aiAverageResponseTime }}
        ms
      </p>
    </section>
  </aside>
</template>
