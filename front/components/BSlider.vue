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
});
const emit = defineEmits(["update:modelValue"]);
const { modelValue: _modelValue } = useVModels(props, emit);
const id = useId();
</script>

<template>
  <div class="flex flex-col gap-4">
    <label :for="id"> {{ label }} </label>
    <div class="flex items-center gap-5">
      <span> {{ _modelValue }} </span>
      <Slider
        v-model="_modelValue"
        :min="0"
        :max="max"
        :step="1"
        class="w-full"
      />
    </div>
  </div>
</template>
