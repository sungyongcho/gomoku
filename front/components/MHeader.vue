<script lang="ts" setup>
import player1Image from "~/assets/player1.webp";
import player2Image from "~/assets/player2.webp";
import aiImage from "~/assets/ai.webp";

const props = defineProps({
  debug: {
    type: Boolean,
    default: false,
  },
  nonGamePage: {
    type: Boolean,
    default: false,
  },
});
const { player1TotalCaptured, player2TotalCaptured, settings, turn } =
  storeToRefs(useGameStore());
const { isAiThinking } = storeToRefs(useGameStore());
</script>

<template>
  <header class="w-full bg-black">
    <div
      class="mx-auto flex max-w-[1280px] items-center -lg:flex-col -sm:gap-3"
    >
      <div class="flex w-full justify-between">
        <nuxt-link
          to="/"
          class="px-4 py-2 text-2xl font-extrabold uppercase text-white -sm:text-sm"
          :class="{ '!text-2xl': nonGamePage }"
        >
          omok
        </nuxt-link>

        <MobileInfoBoard v-if="!nonGamePage" class="lg:hidden" />
      </div>

      <div class="shrink-0 py-2 text-white" v-if="!nonGamePage">
        <section class="flex items-center gap-2 -sm:gap-2 lg:hidden">
          <div class="flex items-center justify-center">
            <button
              :disabled="!debug"
              class="flex flex-col-reverse items-center border-2 border-transparent"
              @click="turn = 'X'"
            >
              <InfoAvatar
                :image="player1Image"
                :loading="isAiThinking && turn === 'X'"
                color="black"
                :active="turn === 'X'"
              />
            </button>

            <div
              class="ml-1 rounded-lg border-2 border-white px-2 py-1 -sm:px-1 -sm:py-0"
              :class="{ ['!border-yellow-500']: turn === 'X' }"
            >
              <span class="text-lg">
                {{ player1TotalCaptured }}
              </span>
              <span class="text-xs text-gray-300">
                / {{ settings.totalPairCaptured }}
              </span>
            </div>
          </div>

          <span class="text-2xl">vs</span>

          <div class="flex items-center justify-center">
            <div
              class="mr-1 rounded-lg border-2 border-white px-2 py-1 -sm:px-1 -sm:py-0"
              :class="{ ['!border-yellow-500']: turn === 'O' }"
            >
              <span class="text-lg">
                {{ player2TotalCaptured }}
              </span>
              <span class="text-xs text-gray-300">
                / {{ settings.totalPairCaptured }}
              </span>
            </div>
            <button
              class="flex flex-col-reverse items-center border-2 border-transparent"
              :disabled="!debug"
              @click="turn = 'O'"
            >
              <InfoAvatar
                :image="settings.isPlayer2AI ? aiImage : player2Image"
                :loading="isAiThinking && turn === 'O'"
                color="white"
                :active="turn === 'O'"
              />
            </button>
          </div>
        </section>
      </div>
    </div>
  </header>
</template>
