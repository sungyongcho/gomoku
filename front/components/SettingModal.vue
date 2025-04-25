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

watch(
  () => settings.value.totalPairCaptured,
  () => {
    settings.value.advantage1 = 0;
    settings.value.advantage2 = 0;
  },
);
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
        <BButtonGroupSwitch
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
            { value: 'X', label: 'black' },
            { value: 'O', label: 'white' },
          ]"
          label="First move"
        />

        <BToggleSwitch
          v-model="settings.enableCapture"
          label="Enable Capture"
        />
        <BToggleSwitch
          v-model="settings.enableDoubleThreeRestriction"
          label="Enable Double-Three Prohibition"
        />
        <BButtonGroupSwitch
          v-if="settings.enableCapture"
          v-model="settings.totalPairCaptured"
          class="col-span-2"
          :options="[
            { value: 3, label: '3' },
            { value: 4, label: '4' },
            { value: 5, label: '5' },
            { value: 6, label: '6' },
            { value: 7, label: '7' },
          ]"
          label="Total of captured pair stones"
        />
        <BSlider
          v-if="settings.enableCapture"
          v-model="settings.advantage1"
          :max="settings.totalPairCaptured - 1"
          label="Player1 Advantage"
        />
        <BSlider
          v-if="settings.enableCapture"
          v-model="settings.advantage2"
          :max="settings.totalPairCaptured - 1"
          label="Player2 Advantage"
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
