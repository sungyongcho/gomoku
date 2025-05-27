<script setup lang="ts">
import { NuxtLink } from "#components";

const { settings } = storeToRefs(useGameStore());
const { initGame } = useGameStore();
const $router = useRouter();
const onGameWithAI = () => {
  settings.value.isPlayer2AI = true;
  initGame();
  $router.push("/game/");
};

const onGameWithHuman = () => {
  settings.value.isPlayer2AI = false;
  initGame();
  $router.push("/game/");
};
</script>

<template>
  <main
    class="flex h-screen items-center justify-center overflow-hidden bg-black"
  >
    <section
      class="flex h-[50vw] max-h-[550px] min-h-[500px] w-[50vw] min-w-[500px] max-w-[550px] flex-col items-center justify-center gap-16 rounded-[50%] bg-white px-4 py-20 shadow-[0_0_10px_5px_gray] -md:gap-4 -sm:min-h-[340px] -sm:min-w-[340px]"
    >
      <h1
        class="uppercase typo-giga-title -md:typo-mega-title -sm:typo-large-title"
      >
        gomoku
      </h1>

      <div class="flex w-full max-w-[220px] flex-col gap-2 -sm:max-w-[180px]">
        <Button
          class="mx-auto w-full text-[15px] -sm:px-3 -sm:py-2 -sm:text-[12px]"
          @click="onGameWithAI"
        >
          <div class="flex w-full items-center justify-between">
            <i class="pi pi-user"></i> Player vs AI
            <i class="pi pi-android"></i>
          </div>
        </Button>
        <Button
          class="mx-auto w-full text-[15px] -sm:px-3 -sm:py-2 -sm:text-[12px]"
          @click="onGameWithHuman"
        >
          <div class="flex w-full items-center justify-between">
            <i class="pi pi-user"></i> Player vs Player
            <i class="pi pi-user"></i>
          </div>
        </Button>

        <Button
          class="mx-auto w-full text-[15px] -sm:px-3 -sm:py-2 -sm:text-[12px]"
          :as="NuxtLink"
          to="/docs/intro"
        >
          <i class="pi pi-question-circle"></i>
          Help
        </Button>

        <Button
          @click="$router.push('/test')"
          class="mx-auto w-full text-[15px] -sm:px-3 -sm:py-2 -sm:text-[12px]"
        >
          <i class="pi pi-clipboard"></i>
          <span> Evaluation Test</span>
        </Button>
      </div>
    </section>
  </main>
</template>
