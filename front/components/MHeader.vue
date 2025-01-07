<script lang="ts" setup>
import player1Image from "~/assets/player1.webp";
import player2Image from "~/assets/player2.webp";
import aiImage from "~/assets/ai.webp";

const { player1TotalCaptured, player2TotalCaptured, settings } =
  storeToRefs(useGameStore());
const { deleteLastHistory, initGame } = useGameStore();
</script>

<template>
  <header class="w-full bg-black p-4">
    <div
      class="mx-auto flex max-w-[1280px] items-center justify-center -lg:justify-between -sm:flex-col -sm:gap-3"
    >
      <nuxt-link to="/" class="text-2xl font-extrabold uppercase text-white">
        omok
      </nuxt-link>

      <div class="text-white">
        <section class="flex items-center gap-4 -sm:gap-2 lg:hidden">
          <div class="flex items-center justify-center gap-2">
            <div class="flex flex-col-reverse items-center">
              <span class="text-sm">Player1</span>
              <Avatar
                :image="player1Image"
                shape="circle"
                class="border-2 border-black shadow-[0_0_4px_1px_white]"
              />
            </div>

            <div class="ml-2 rounded-lg border-2 border-white px-2 py-1">
              <span class="text-lg">
                {{ player1TotalCaptured }}
              </span>
              <span class="text-xs text-gray-300">
                / {{ settings.totalPairCaptured }}
              </span>
            </div>
          </div>

          <span class="text-2xl">vs</span>

          <div class="flex items-center justify-center gap-2">
            <div class="mr-2 rounded-lg border-2 border-white px-2 py-1">
              <span class="text-lg">
                {{ player2TotalCaptured }}
              </span>
              <span class="text-xs text-gray-300">
                / {{ settings.totalPairCaptured }}
              </span>
            </div>
            <div class="flex flex-col-reverse items-center">
              <span class="text-sm">{{
                settings.isPlayer2AI ? "AI" : "Player2"
              }}</span>
              <Avatar
                :image="settings.isPlayer2AI ? aiImage : player2Image"
                shape="circle"
                class="border-2 border-white shadow-[0_0_8px_1px_gray]"
              />
            </div>
          </div>
        </section>
      </div>
    </div>
  </header>
</template>
