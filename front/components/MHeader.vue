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
  testPage: {
    type: Boolean,
    default: false,
  },
});
const {
  player1TotalCaptured,
  player2TotalCaptured,
  settings,
  turn,
  showSettings,
} = storeToRefs(useGameStore());
const { initGame } = useGameStore();
const { isAiThinking } = storeToRefs(useGameStore());
const $router = useRouter();
const $route = useRoute();
const { isProd } = useEnv();

const onGameDebug = () => {
  settings.value.isPlayer2AI = false;
  initGame();
  $router.push("/debug");
};
</script>

<template>
  <header class="w-full bg-black px-4 py-2 -sm:px-0">
    <div class="mx-auto flex max-w-[1140px] items-center -lg:flex-col">
      <div class="flex w-full shrink-0 justify-between">
        <nuxt-link
          to="/"
          class="flex items-center px-4 py-2 text-xl font-extrabold uppercase text-white"
        >
          gomoku
        </nuxt-link>

        <div class="mr-2 flex items-center gap-1 -sm:my-0">
          <Button
            icon="pi pi-cog"
            variant="text"
            rounded
            @click="showSettings = true"
            class="!text-white hover:!text-black"
          />
          <MobileInfoBoard v-if="!nonGamePage" class="lg:hidden" />

          <Button
            icon="pi pi-wrench"
            rounded
            variant="text"
            class="!text-white hover:!text-black"
            @click="onGameDebug"
            v-if="$route.path === '/' && !isProd"
          />
        </div>
      </div>

      <div class="shrink-0 py-2 text-white" v-if="!testPage && !nonGamePage">
        <section class="flex items-center gap-2 -sm:gap-2 lg:hidden">
          <div class="flex items-center justify-center">
            <button
              :disabled="!debug"
              class="flex flex-col-reverse items-center border-2 border-transparent"
              @click="
                settings.firstMove === 'Player1' ? (turn = 'X') : (turn = 'O')
              "
            >
              <InfoAvatar
                :image="player1Image"
                :loading="
                  isAiThinking &&
                  ((settings.firstMove === 'Player1' && turn === 'X') ||
                    (settings.firstMove === 'Player2' && turn === 'O'))
                "
                :color="settings.firstMove === 'Player1' ? 'black' : 'white'"
                :active="
                  (settings.firstMove === 'Player1' && turn === 'X') ||
                  (settings.firstMove === 'Player2' && turn === 'O')
                "
              />
            </button>

            <div
              class="ml-1 rounded-lg border-2 border-white px-2 py-1 -sm:px-1 -sm:py-0"
              :class="{
                ['!border-yellow-500']:
                  settings.firstMove == 'Player1' ? turn === 'X' : turn === 'O',
              }"
              v-if="settings.enableCapture"
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
              :class="{
                ['!border-yellow-500']:
                  settings.firstMove == 'Player2' ? turn === 'X' : turn === 'O',
              }"
              v-if="settings.enableCapture"
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
              @click="
                settings.firstMove === 'Player2' ? (turn = 'X') : (turn = 'O')
              "
            >
              <InfoAvatar
                :image="settings.isPlayer2AI ? aiImage : player2Image"
                :loading="
                  isAiThinking &&
                  ((settings.firstMove === 'Player1' && turn === 'O') ||
                    (settings.firstMove === 'Player2' && turn === 'X'))
                "
                :color="settings.firstMove === 'Player2' ? 'black' : 'white'"
                :active="
                  (settings.firstMove === 'Player1' && turn === 'O') ||
                  (settings.firstMove === 'Player2' && turn === 'X')
                "
              />
            </button>
          </div>
        </section>
      </div>
    </div>
  </header>
</template>
