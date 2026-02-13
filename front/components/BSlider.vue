<script setup lang="ts">
import { useVModels } from "@vueuse/core";

const props = defineProps({
  modelValue: {
    type: Number,
    default: 0,
  },
  label: {
    type: String,
    default: "",
  },
  name: {
    type: String,
    default: "",
  },
  max: {
    type: Number,
    default: 3,
  },
  disabled: {
    type: Boolean,
    default: false,
  },
});
const emit = defineEmits(["update:modelValue"]);
const { modelValue: _modelValue } = useVModels(props, emit);
const id = useId();
</script>

<template>
  <div
    class="flex flex-col gap-4 transition-opacity"
    :class="{
      'opacity-60 cursor-not-allowed pointer-events-none select-none': disabled,
    }"
  >
    <label :for="id" class="transition-colors" :class="{ 'text-gray-500': disabled }">
      {{ label }}
    </label>
    <div class="flex items-center gap-5">
      <span :class="{ 'text-gray-500': disabled }"> {{ _modelValue }} </span>
      <Slider
        v-model="_modelValue"
        :min="0"
        :max="max"
        :step="1"
        class="w-full"
        :disabled="disabled"
      />
    </div>
  </div>
</template>
