<script setup lang="ts">
const showSettings = ref(false);
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

const onGameDebug = () => {
  settings.value.isPlayer2AI = false;
  initGame();
  $router.push("/debug");
};
</script>

<template>
  <main class="flex h-screen items-center justify-center bg-black">
    <section
      class="flex w-full max-w-[520px] flex-col items-center justify-center gap-20 rounded-[50%] bg-white px-4 py-20 shadow-[0_0_10px_5px_gray]"
    >
      <h1 class="uppercase typo-giga-title">Omok</h1>

      <div class="flex flex-col gap-4">
        <Button class="w-[220px]" size="large" @click="onGameWithAI">
          Player vs AI
        </Button>
        <Button class="w-[220px]" size="large" @click="onGameWithHuman"
          >Player vs Player</Button
        >
        <Button @click="showSettings = true" class="w-[220px]" size="large">
          <i class="pi pi-cog text-[20px]"></i> Settings
        </Button>
      </div>
    </section>

    <SettingModal v-model:visible="showSettings" />

    <button
      class="absolute right-[30px] top-[30px] flex items-center justify-center rounded-[50%] bg-white p-3 transition-colors hover:bg-gray-200"
      @click="onGameDebug"
    >
      <i class="pi pi-wrench text-[20px]"></i>
    </button>
  </main>
</template>
