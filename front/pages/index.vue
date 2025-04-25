<script setup lang="ts">
const { settings } = storeToRefs(useGameStore());
const { initGame } = useGameStore();
const $router = useRouter();
const onGameWithAI = () => {
  settings.value.isPlayer2AI = true;
  initGame();
  $router.push("/game");
};

const onGameWithHuman = () => {
  settings.value.isPlayer2AI = false;
  initGame();
  $router.push("/game");
};

const onHowToPlay = () => {
  $router.push("/how-to-play");
};
</script>

<template>
  <main
    class="flex h-[calc(100vh-72px)] items-center justify-center overflow-hidden bg-black"
  >
    <section
      class="flex h-[50vw] min-h-[500px] w-[50vw] min-w-[500px] flex-col items-center justify-center gap-20 rounded-[50%] bg-white px-4 py-20 shadow-[0_0_10px_5px_gray]"
    >
      <h1 class="uppercase typo-giga-title">Omok</h1>

      <div class="flex flex-col gap-4">
        <Button class="w-[220px]" size="large" @click="onGameWithAI">
          <div class="flex w-full items-center justify-between">
            <i class="pi pi-user text-[20px]"></i> Player vs AI
            <i class="pi pi-android text-[20px]"></i>
          </div>
        </Button>
        <Button class="w-[220px]" size="large" @click="onGameWithHuman">
          <div class="flex w-full items-center justify-between">
            <i class="pi pi-user text-[20px]"></i> Player vs Player
            <i class="pi pi-user text-[20px]"></i>
          </div>
        </Button>

        <Button @click="onHowToPlay" class="w-[220px]" size="large">
          <i class="pi pi-question-circle text-[20px]"></i> How to play
        </Button>

        <Button @click="$router.push('/test')" class="w-[220px]" size="large">
          <i class="pi pi-clipboard text-[20px]"></i> Evaluation Test
        </Button>
      </div>
    </section>
  </main>
</template>
