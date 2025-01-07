<script setup lang="ts">
import player1Image from "~/assets/player1.webp";
import player2Image from "~/assets/player2.webp";
import aiImage from "~/assets/ai.webp";

const { histories, player1TotalCaptured, player2TotalCaptured, settings } =
  storeToRefs(useGameStore());
const { historyToLog } = useGameStore();
const historyEl = ref<HTMLElement>();

watch(histories.value, () => {
  nextTick(() => {
    const scrollHeight = historyEl.value!.scrollHeight;
    historyEl.value!.scroll(0, scrollHeight);
  });
});
</script>
<template>
  <aside class="flex w-[300px] flex-col items-center gap-5">
    <section class="flex w-full items-center justify-between gap-10 -lg:hidden">
      <div class="flex flex-col items-center justify-center">
        <Avatar
          size="xlarge"
          shape="circle"
          :image="player1Image"
          class="border-4 border-black shadow-[0_0_4px_1px_gray]"
        />
        <span>Player1</span>
      </div>

      <span class="text-2xl">vs</span>

      <div class="flex flex-col items-center justify-center">
        <Avatar
          class="border-4 border-gray-100 shadow-[0_0_4px_1px_black]"
          size="xlarge"
          shape="circle"
          :image="settings.isPlayer2AI ? aiImage : player2Image"
        />
        <span> {{ settings.isPlayer2AI ? "AI" : "Player2" }}</span>
      </div>
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
        class="h-[50vh] w-full overflow-y-auto rounded-md bg-black px-4 py-2 text-sm"
      >
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
