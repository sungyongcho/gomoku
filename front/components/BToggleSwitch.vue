<script setup lang="ts">
import { useVModels } from "@vueuse/core";

const props = defineProps({
  modelValue: {
    type: Boolean,
    default: false,
  },
  label: {
    type: String,
    default: "",
  },
  name: {
    type: String,
    default: "",
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
    class="flex flex-col gap-2 transition-opacity"
    :class="{
      'opacity-60 cursor-not-allowed pointer-events-none select-none': disabled,
    }"
  >
    <label :for="id" :class="{ 'text-gray-500': disabled }"> {{ label }} </label>
    <ToggleSwitch :id="id" :name="name" v-model="_modelValue" :disabled="disabled" />
  </div>
</template>
