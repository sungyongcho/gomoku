<script setup lang="ts">
import { useVModels } from "@vueuse/core";

const props = defineProps({
  visible: {
    type: Boolean,
    default: false,
  },
});
const emit = defineEmits(["update:visible"]);
const { visible: _visible } = useVModels(props, emit);
const { settings } = storeToRefs(useGameStore());
const onSave = () => {
  _visible.value = false;
};

const isAlphaZero = computed(() => settings.value.ai === "alphazero");
const docsAlphaZeroUrl = "/docs/alphazero/intro";

watch(
  () => settings.value.totalPairCaptured,
  () => {
    settings.value.advantage1 = 0;
    settings.value.advantage2 = 0;
  },
);

watch(isAlphaZero, (v) => {
  if (v) {
    settings.value.enableDoubleThreeRestriction = true;
    settings.value.enableCapture = true;
    settings.value.totalPairCaptured = 5;
    settings.value.advantage1 = 0;
    settings.value.advantage2 = 0;
  }
}, { immediate: true });
</script>

<template>
  <Dialog
    v-model:visible="_visible"
    modal
    dismissableMask
    header="Settings"
    :style="{ width: '90vw', maxWidth: '800px' }"
    class="bg-white"
    closeOnEscape
    pt:mask:class="backdrop-blur-sm"
  >
    <section class="flex flex-col items-center justify-center">
      <form class="mb-[60px] grid w-full grid-cols-2 gap-8">
        <div :class="{ 'col-span-2': settings.ai === 'alphazero' }">
          <BButtonGroupSwitch
            v-model="settings.ai"
            :options="[
              { value: 'minimax', label: 'minimax' },
              { value: 'alphazero', label: 'alphazero' },
            ]"
            label="AI model"
          />
          <div
            v-if="settings.ai === 'alphazero'"
            class="mt-4 rounded-lg border border-blue-200 bg-blue-50 p-4 text-sm text-blue-900"
          >
            <p class="mb-2 font-medium">AlphaZero mode: 200 MCTS simulations per move. During training, this was increased to 2,400.</p>
            <NuxtLink
              :to="docsAlphaZeroUrl"
              class="font-medium underline hover:no-underline"
              @click="_visible = false"
            >
              See more details in docs
            </NuxtLink>
          </div>
        </div>

        <BButtonGroupSwitch
          v-if="settings.ai === 'minimax'"
          v-model="settings.difficulty"
          :options="[
            { value: 'easy', label: 'easy' },
            { value: 'medium', label: 'medium' },
            { value: 'hard', label: 'hard' },
          ]"
          label="Difficulty"
        />


        <BButtonGroupSwitch
          v-model="settings.firstMove"
          :options="[
            { value: 'Player1', label: 'Player1' },
            { value: 'Player2', label: 'Player2' },
          ]"
          label="First move"
        />

        <BToggleSwitch
          v-model="settings.enableDoubleThreeRestriction"
          label="Enable Double-Three Prohibition"
          :disabled="isAlphaZero"
        />

        <BToggleSwitch
          v-model="settings.enableCapture"
          label="Enable Capture"
          :disabled="isAlphaZero"
        />

        <BButtonGroupSwitch
          v-if="settings.enableCapture"
          v-model="settings.totalPairCaptured"
          :options="[
            { value: 3, label: '3' },
            { value: 4, label: '4' },
            { value: 5, label: '5' },
            { value: 6, label: '6' },
            { value: 7, label: '7' },
          ]"
          label="Total of captured pair stones"
          :disabled="isAlphaZero"
        />
        <BSlider
          v-if="settings.enableCapture"
          v-model="settings.advantage1"
          :max="settings.totalPairCaptured - 1"
          label="Player1 Advantage"
          :disabled="isAlphaZero"
        />
        <BSlider
          v-if="settings.enableCapture"
          v-model="settings.advantage2"
          :max="settings.totalPairCaptured - 1"
          label="Player2 Advantage"
          :disabled="isAlphaZero"
        />
      </form>

      <div class="flex w-full gap-4">
        <Button @click="onSave" class="flex-1">
          <i class="pi pi-save text-[20px]"></i>Save
        </Button>
        <Button @click="_visible = false" severity="secondary" class="flex-1">
          <i class="pi pi-times text-[20px]"></i>Close
        </Button>
      </div>
    </section>
  </Dialog>
</template>
